import os
import logging
from datetime import timedelta
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
import torch.distributed as dist

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from ema_pytorch import EMA
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.utils.versions import require_version

from peft import LoraConfig, get_peft_model

from galaxea_fm.data.base_lerobot_dataset import BaseLerobotDataset
from galaxea_fm.processors.base_processor import BaseProcessor
from galaxea_fm.models.base_policy import BasePolicy
from galaxea_fm.models.galaxea_zero.galaxea_zero_policy import GalaxeaZeroPolicy
from galaxea_fm.utils.get_scheduler import get_scheduler
from galaxea_fm.utils.logging_config import (
    setup_logging,
    log_allocated_gpu_memory,
    log_amp_config,
)
from galaxea_fm.utils.pytorch_utils import set_global_seed
from galaxea_fm.utils.dist import ResumableDistributedSampler
from galaxea_fm.utils.train_utils import MFUTracker, init_experiment_tracker, register_graceful_exit
from galaxea_fm.utils.normalizer import (
    load_dataset_stats_from_json, 
    save_dataset_stats_to_json, 
    search_dataset_stats_cache_json, 
)
from galaxea_fm.utils.config_resolvers import register_default_resolvers
from galaxea_fm.utils.train_utils import set_global_monitor, get_global_monitor
from galaxea_fm.utils.tqdm import tqdm
from galaxea_fm.utils.git_info import save_git_info, GitInfoError
from galaxea_fm.utils.load_pretrained_resumed import save_checkpoint, load_embedded_dataset_stats

register_default_resolvers()
logger = get_logger(__name__)
require_version("datasets==3.6.0", "To fix: uv pip install datasets==3.6.0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_base_weights_into_model(pretrained_path, model):
    pretrained_path = Path(pretrained_path)
    pretrained_dict = torch.load(pretrained_path / "model.pt", weights_only=True, map_location='cpu')
    model_dict = model.state_dict()

    unexpected_keys, mismatched_key_shapes, match_key_tensors = [], {}, {}
    for key, ckpt_param in pretrained_dict.items():
        if key not in model_dict:
            unexpected_keys.append(key)
        elif model_dict[key].shape != ckpt_param.shape:
            mismatched_key_shapes[key] = (model_dict[key].shape, ckpt_param.shape)
        else:
            match_key_tensors[key] = ckpt_param

    incompatible = model.load_state_dict(match_key_tensors, strict=False)
    assert len(incompatible.unexpected_keys) == 0
    missing_keys = list(set(incompatible.missing_keys) - set(mismatched_key_shapes.keys()))

    logger.info(f"Loaded {len(match_key_tensors)} / {len(model_dict)} keys from pretrained checkpoint")
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)}")
        for k in missing_keys:
            logger.warning(f"  {k}")
    if mismatched_key_shapes:
        logger.warning(f"Shape mismatched keys: {len(mismatched_key_shapes)}")
        for k, (model_shape, ckpt_shape) in mismatched_key_shapes.items():
            logger.warning(f"  {k}: model {model_shape}, ckpt {ckpt_shape}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys from checkpoint: {len(unexpected_keys)}")
        for k in unexpected_keys:
            logger.warning(f"  {k}")


def inject_lora_into_model(model, lora_cfg):
    modules_to_save = list(lora_cfg.get("modules_to_save", [])) if lora_cfg.get("modules_to_save") else None
    lora_config = LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.dropout,
        init_lora_weights=lora_cfg.init_lora_weights,
        modules_to_save=modules_to_save,
    )
    logger.info(f"Injecting LoRA adapters: rank={lora_cfg.rank}, alpha={lora_cfg.alpha}, "
                f"target_modules={lora_cfg.target_modules}, dropout={lora_cfg.dropout}, "
                f"modules_to_save={modules_to_save}")
    model.model = get_peft_model(model.model, lora_config)
    return model


def freeze_non_lora_params(model):
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    logger.info(f"Frozen {frozen_count} non-LoRA params, kept {trainable_count} LoRA/modules_to_save params trainable")


def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} = {100*trainable/total:.2f}%")


def get_lora_param_groups(model, lr, weight_decay, vlm_lr_multiplier):
    vlm_lora_params = []
    action_expert_params = []

    action_expert_keywords = [
        "action_encoder", "action_decoder", "proprio_encoder",
        "mixtures.action", "time_embedding",
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in action_expert_keywords):
            action_expert_params.append(param)
        else:
            vlm_lora_params.append(param)

    param_groups = [
        {
            "params": vlm_lora_params,
            "lr": lr * vlm_lr_multiplier,
            "weight_decay": weight_decay,
            "name": "vlm_lora",
        },
        {
            "params": action_expert_params,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": "action_expert_lora",
        },
    ]

    all_requires_grad = [p for p in model.parameters() if p.requires_grad]
    num_grouped = sum(len(g['params']) for g in param_groups)
    assert len(all_requires_grad) == num_grouped, \
        f"Param group mismatch: {len(all_requires_grad)} trainable vs {num_grouped} grouped"

    logger.info(f"VLM LoRA params: {len(vlm_lora_params)}, "
                f"Action Expert params (LoRA + modules_to_save): {len(action_expert_params)}")
    return param_groups


def save_lora_adapter(model, save_path):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, 'model') and hasattr(model.model, 'save_pretrained'):
        model.model.save_pretrained(save_path)
        logger.info(f"LoRA adapter saved to {save_path}")
    else:
        logger.warning("Model does not have peft save_pretrained method, skipping adapter save")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def finetune_lora(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    output_dir = Path(cfg.output_dir)

    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    project_config = ProjectConfiguration(project_dir=str(Path(cfg.output_dir)))
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.model.enable_bf16_training else "no",
        project_config=project_config,
        kwargs_handlers=[init_process_group_kwargs],
        log_with=cfg.logger.type,
    )
    register_graceful_exit(accelerator)
    torch.cuda.set_device(device_id := accelerator.local_process_index)
    torch.cuda.empty_cache()

    setup_logging(log_level=logging.INFO, is_main_process=accelerator.is_main_process)
    logger.info(f"Output directory: {output_dir}")
    log_amp_config(logger, accelerator)
    init_experiment_tracker(cfg, accelerator, output_dir)
    set_global_monitor()
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)

    if accelerator.is_main_process:
        try:
            git_info_path = save_git_info(output_dir=output_dir)
            logger.info(f"Git info saved to: {git_info_path}")
        except GitInfoError as e:
            logger.warning(f"Could not save git info: {e}")

    # ========== Step 1: Create model ==========
    model: BasePolicy = instantiate(cfg.model.model_arch)

    # ========== Step 2: Load base pretrained weights (BEFORE LoRA injection) ==========
    if not cfg.resume_ckpt and cfg.model.pretrained_ckpt:
        logger.info(f"Loading base pretrained weights from {cfg.model.pretrained_ckpt}")
        load_base_weights_into_model(cfg.model.pretrained_ckpt, model)

    # ========== Step 3: Inject LoRA ==========
    lora_cfg = cfg.lora
    model = inject_lora_into_model(model, lora_cfg)

    # ========== Step 4: Freeze all non-LoRA parameters ==========
    freeze_non_lora_params(model)
    print_trainable_params(model)

    # ========== Step 5: bf16 / EMA / SyncBN / compile ==========
    if cfg.model.model_weights_to_bf16:
        model = model.to(torch.bfloat16)

    use_ema = cfg.model.use_ema
    if use_ema:
        ema_model = EMA(
            model,
            update_after_step=cfg.model.ema.update_after_step,
            beta=cfg.model.ema.power,
        ).to(device_id)
    else:
        ema_model = None

    if cfg.model.use_sync_bn and accelerator.num_processes > 1:
        logger.info("Use sync batch norm.")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if cfg.model.use_torch_compile:
        model = torch.compile(model, mode="default")

    model = model.to(device_id)
    log_allocated_gpu_memory(logger, stage="loading model", device=0)

    # ========== Step 6: Dataset & Processor ==========
    train_dataset: BaseLerobotDataset = instantiate(cfg.data.dataset, is_training_set=True)
    eval_dataset: BaseLerobotDataset = instantiate(cfg.data.dataset, is_training_set=False)
    train_processor: BaseProcessor = instantiate(cfg.data.processor)
    eval_processor: BaseProcessor = instantiate(cfg.data.processor)
    train_dataset.set_processor(train_processor)
    eval_dataset.set_processor(eval_processor)

    train_sampler = ResumableDistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        batch_size=cfg.model.batch_size,
    )
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=cfg.model.num_workers,
        pin_memory=cfg.model.pin_memory,
        persistent_workers=cfg.model.persistent_workers,
        worker_init_fn=worker_init_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size_val,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=cfg.model.num_workers,
        pin_memory=cfg.model.pin_memory,
        persistent_workers=cfg.model.persistent_workers,
        worker_init_fn=worker_init_fn,
    )

    if cfg.model.max_epochs:
        assert not cfg.model.max_steps, "Cannot set both `max_epochs` and `max_steps`!"
        steps_per_epoch = len(train_dataloader) // cfg.model.grad_accumulation_steps
        max_steps = steps_per_epoch * cfg.model.max_epochs
    else:
        max_steps = cfg.model.max_steps

    use_mfu_tracker = isinstance(model, GalaxeaZeroPolicy)

    # ========== Step 7: DDP wrap ==========
    model = DDP(model, device_ids=[device_id], find_unused_parameters=cfg.model.find_unused_parameters, gradient_as_bucket_view=True)

    # ========== Step 8: Optimizer with LoRA param groups ==========
    param_groups = get_lora_param_groups(
        model,
        lr=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        vlm_lr_multiplier=lora_cfg.vlm_lr_multiplier,
    )
    betas = tuple(cfg.model.betas)
    if cfg.model.use_8bit_optimizer:
        assert bnb is not None, "bitsandbytes is not installed, cannot use 8bit optimizer"
        optimizer = bnb.optim.AdamW8bit(param_groups, betas=betas)
    else:
        optimizer = AdamW(param_groups, betas=betas)

    if cfg.model.lr_scheduler_type == "OneCycleLR":
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=cfg.model.learning_rate,
            total_steps=max_steps,
            pct_start=cfg.model.pct_start,
            anneal_strategy=cfg.model.anneal_strategy,
            div_factor=cfg.model.div_factor,
            final_div_factor=cfg.model.final_div_factor,
        )
    else:
        scheduler = get_scheduler(
            name=cfg.model.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=cfg.model.warmup_steps,
            num_training_steps=max_steps,
        )

    # ========== Step 9: Resume or compute dataset stats ==========
    if cfg.resume_ckpt:
        resume_dataloader = True
        from galaxea_fm.utils.load_pretrained_resumed import resume_checkpoint
        step, epoch, batch_idx = resume_checkpoint(
            checkpoint_path=cfg.resume_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_model=ema_model,
            device_id=device_id,
        )
        dataset_stats = load_embedded_dataset_stats(cfg.resume_ckpt)
        logger.info(f"Resume training from step {step}, epoch {epoch}, batch_idx {batch_idx}")
    else:
        resume_dataloader = False
        step, epoch, batch_idx = 0, 0, 0

        if cfg.model.pretrained_ckpt and cfg.model.use_pretrained_norm_stats:
            logger.info(f"Use pretrained dataset stats from {cfg.model.pretrained_ckpt}")
            dataset_stats = load_embedded_dataset_stats(cfg.model.pretrained_ckpt)
        else:
            logger.info("Calculate stats from dataset instead of loading from pretrained")
            if accelerator.is_main_process:
                exist_cache, cache_path = search_dataset_stats_cache_json(cfg.dataset_stats_cache_dir, cfg.data)
                if exist_cache:
                    logger.info(f"  Use dataset stats cache file {cache_path}")
                    dataset_stats = load_dataset_stats_from_json(cache_path)
                else:
                    logger.info("  No cached stats found, computing from dataset ...")
                    dataset_stats = train_dataset.get_dataset_stats(train_processor)
                    save_dataset_stats_to_json(dataset_stats, cache_path)
                    logger.info(f"  Saved dataset stats cache: {cache_path}")
            else:
                dataset_stats = None

            container = [dataset_stats]
            dist.broadcast_object_list(container, src=0)
            dataset_stats = container[0]

    train_processor.set_normalizer_from_stats(dataset_stats)
    eval_processor.set_normalizer_from_stats(dataset_stats)
    if accelerator.is_main_process:
        save_dataset_stats_to_json(dataset_stats, output_dir / "dataset_stats.json")

    # ========== Step 10: MFU Tracker ==========
    mfu_tracker = None
    if accelerator.is_main_process:
        effective_batch_size = cfg.model.batch_size * cfg.model.grad_accumulation_steps * dist.get_world_size()
        if use_mfu_tracker:
            mfu_tracker = MFUTracker(
                model=model.module,
                batch_size=effective_batch_size,
                device_id=device_id,
                update_interval=cfg.logger.log_steps,
                world_size=dist.get_world_size(),
                dtype=torch.bfloat16 if cfg.model.enable_bf16_training else torch.float32,
            )
            mfu_tracker.reset(step)

    accelerator.wait_for_everyone()

    # ========== Step 11: Training loop ==========
    training_done = False
    with tqdm.tqdm(initial=step, total=max_steps, leave=False, dynamic_ncols=True) as progress:
        while not training_done:
            train_sampler.set_epoch(epoch)
            if resume_dataloader:
                logger.info(f"Resume dataloader state from batch_idx {batch_idx} of epoch {epoch}")
                train_sampler.set_start_batch(batch_idx)
                resume_dataloader = False
            else:
                batch_idx = 0
                train_sampler.set_start_batch(0)

            data_iter = iter(train_dataloader)
            model.train()
            optimizer.zero_grad()
            while batch_idx < len(train_dataloader):
                batch = next(data_iter)
                is_optimizer_step = (batch_idx + 1) % cfg.model.grad_accumulation_steps == 0
                sync_ctx = model.no_sync() if not is_optimizer_step else nullcontext()
                with sync_ctx:
                    with accelerator.autocast():
                        loss, loss_value_dict = model(batch)
                    normalized_loss = loss / cfg.model.grad_accumulation_steps
                    normalized_loss.backward()

                batch_idx += 1

                if is_optimizer_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    progress.set_description(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                    progress.update()
                    progress.refresh()

                    if use_ema:
                        ema_model.update()

                    step += 1

                    if step % cfg.logger.log_steps == 0:
                        log_dict = {k: (v.item() if hasattr(v, "item") else float(v)) for k, v in loss_value_dict.items()}
                        log_dict.update({
                            "lr/vlm_lora": optimizer.param_groups[0]["lr"],
                            "lr/action_expert_lora": optimizer.param_groups[1]["lr"],
                            "grad_norm": grad_norm.item(),
                        })

                        if mfu_tracker is not None:
                            mfu_metrics = mfu_tracker.compute_metrics(step)
                            log_dict.update(mfu_metrics)
                        global_monitor = get_global_monitor()
                        if global_monitor is not None:
                            log_dict.update(global_monitor.get_metrics())

                        accelerator.log(log_dict, step=step)

                # Save checkpoint
                if step > 0 and (step % cfg.checkpointing_steps) == 0:
                    if accelerator.is_main_process:
                        logger.info(f"Saving model checkpoint for step {step} ...")
                        unwrapped_model = accelerator.unwrap_model(model)
                        checkpoint_path = output_dir / "checkpoints" / f"step_{step}"
                        save_checkpoint(
                            path=checkpoint_path,
                            step=step,
                            epoch=epoch,
                            batch_idx=batch_idx,
                            model=unwrapped_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            ema_model=ema_model,
                            dataset_stats=dataset_stats,
                            cfg=cfg,
                        )
                        save_lora_adapter(unwrapped_model, checkpoint_path / "lora_adapter")

                    accelerator.wait_for_everyone()

                if step >= max_steps:
                    logger.info(f"Max step {max_steps} reached, stop training ...")
                    training_done = True
                    break

            epoch += 1

    # Save final checkpoint
    if accelerator.is_main_process:
        logger.info(f"Saving final model checkpoint for step {step} ...")
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint_path = output_dir / "checkpoints" / f"step_{step}"
        save_checkpoint(
            path=checkpoint_path,
            step=step,
            epoch=epoch,
            batch_idx=batch_idx,
            model=unwrapped_model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_model=ema_model,
            dataset_stats=dataset_stats,
            cfg=cfg,
        )
        save_lora_adapter(unwrapped_model, checkpoint_path / "lora_adapter")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    finetune_lora()
