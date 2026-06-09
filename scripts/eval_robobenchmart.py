#!/usr/bin/env python3
"""Closed-loop evaluation on RoboBenchMart environments.

Directly loads a LoRA-finetuned G0Plus policy (no websocket / port needed),
runs N episodes, and reports per-scene + aggregate success rates.

Usage:
    python scripts/eval_robobenchmart.py \
        task=robobenchmart/fetch_lora_finetune \
        +ckpt_path=./ckpts/fetch_lora_best/model.pt \
        +eval_scene_dir=$RBM_ROOT/demo_envs/pick_to_basket \
        +eval_env_name=PickToBasketContNiveaEnv \
        +eval_num_traj=10 +eval_save_video=true

Specify GPU:
    EVAL_GPU=0 python scripts/eval_robobenchmart.py ...
    # or
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_robobenchmart.py ...
"""

import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EvalRoboBenchMart")

RBM_ROOT = os.environ.get("RBM_ROOT", "/home/lh/VLA/RoboBenchMart-main")

# Fetch pd_joint_pos action layout (13-DoF):
#   [0-6]   arm: shoulder_pan, shoulder_lift, upperarm_roll,
#           elbow_flex, forearm_roll, wrist_flex, wrist_roll
#   [7]     gripper: mimic (1 value controls both fingers)
#   [8-10]  body: head_pan, head_tilt, torso_lift
#   [11-12] base: forward_vel, rotation_vel
FETCH_ACTION_DIM = 13


# ------------------------------------------------------------------
# 1. Model loading
# ------------------------------------------------------------------

def load_policy(ckpt_path: str, cfg: DictConfig, device: str = "cuda"):
    """Load LoRA-finetuned G0Plus policy + processor from a resolved Hydra cfg."""
    from hydra.utils import instantiate
    from galaxea_fm.utils.load_pretrained_resumed import (
        load_checkpoint_for_eval,
        load_dataset_stats_from_json,
    )

    # Skip loading pre-trained VLM weights during eval — the full checkpoint
    # (including LoRA-adapted weights) will be loaded below.
    # If pretrained_model_path is a placeholder, instantiate would fail with
    # "No pre-trained weights found".
    try:
        OmegaConf.set_struct(cfg.model.model_arch, False)
        cfg.model.model_arch.pretrained_model_path = None
        OmegaConf.set_struct(cfg.model.model_arch, True)
    except Exception:
        pass

    policy = instantiate(cfg.model.model_arch)

    # Load checkpoint — handle both directory format and legacy .pt format.
    # Legacy .pt may contain {"model_state_dict": ..., ...} OR be a bare state_dict.
    ckpt = Path(ckpt_path)
    if ckpt.is_dir():
        # New format: directory with model.pt + dataset_stats.json
        policy, stats = load_checkpoint_for_eval(ckpt_path, policy, device="cpu")
    else:
        # Legacy format: single .pt file
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            policy.load_state_dict(state_dict["model_state_dict"], strict=True)
        else:
            # Bare state_dict (no wrapper)
            policy.load_state_dict(state_dict, strict=True)
        # Find dataset_stats.json: same location as new-format checkpoints
        stats_path = ckpt.parent.parent / "dataset_stats.json"
        if not stats_path.exists():
            # Try other common locations
            for candidate in [
                ckpt.parent / "dataset_stats.json",
                ckpt.parent.parent.parent / "dataset_stats.json",
            ]:
                if candidate.exists():
                    stats_path = candidate
                    break
        stats = load_dataset_stats_from_json(stats_path)

    policy = policy.to(device).eval()

    # Move action tokenizer to same device if present
    if hasattr(policy, 'action_tokenizer'):
        policy.action_tokenizer.to(device)

    processor = instantiate(cfg.data.processor)
    processor.set_normalizer_from_stats(stats)
    processor.eval()

    # Set tokenizer for autoregressive models
    if hasattr(policy, 'set_tokenizer') and hasattr(processor, 'tokenizer'):
        policy.set_tokenizer(processor.tokenizer)

    logger.info(f"Loaded policy from {ckpt_path}")
    return policy, processor


# ------------------------------------------------------------------
# 2. Observation construction
# ------------------------------------------------------------------

def get_sensor_names(obs):
    """Auto-detect sensor camera names from ManiSkill observation.

    ManiSkill returns obs['sensor_data'] as a dict of camera names.
    For Fetch: left_base_camera_link (head), right_base_camera_link, fetch_hand (wrist).
    """
    try:
        all_keys = list(obs.get("sensor_data", {}).keys())
    except Exception:
        all_keys = []

    names = {"head_camera": None, "left_wrist_camera": None, "right_wrist_camera": None}
    for k in all_keys:
        kl = k.lower()
        if "left_base" in kl or "head" in kl or "agent" in kl or "top" in kl:
            names["head_camera"] = k
        elif "fetch_hand" in kl or "wrist" in kl or "hand" in kl:
            names["left_wrist_camera"] = k
            names["right_wrist_camera"] = k  # Fetch has one gripper
        elif "right_base" in kl:
            names["right_wrist_camera"] = k

    # Fallback: try known Fetch camera names
    if names["head_camera"] is None and "left_base_camera_link" in all_keys:
        names["head_camera"] = "left_base_camera_link"
    if names["left_wrist_camera"] is None and "fetch_hand" in all_keys:
        names["left_wrist_camera"] = "fetch_hand"
    if names["right_wrist_camera"] is None and "right_base_camera_link" in all_keys:
        names["right_wrist_camera"] = "right_base_camera_link"
    elif names["right_wrist_camera"] is None:
        names["right_wrist_camera"] = names["left_wrist_camera"]

    return names


def build_vla_obs(obs, sensor_names: dict, image_size: int = 224) -> Dict:
    """Build observation dict for VLA policy from ManiSkill obs.

    Args:
        obs: observation dict from env.step()/env.reset(), with 'sensor_data' and 'agent' keys.
        sensor_names: mapping from VLA camera role to ManiSkill sensor key.
        image_size: resize images to (image_size, image_size).

    Returns images as float32 [0,1] tensors in (C, H, W) format,
    matching what the processor expects after ToTensor.
    """
    import cv2

    sensor_data = obs.get("sensor_data", {})

    head_rgb_raw = None
    left_wrist_rgb_raw = None

    head_key = sensor_names.get("head_camera")
    if head_key and head_key in sensor_data:
        try:
            head_rgb_raw = sensor_data[head_key]["rgb"][0].cpu().numpy()
        except Exception:
            pass

    left_key = sensor_names.get("left_wrist_camera")
    if left_key and left_key in sensor_data:
        try:
            left_wrist_rgb_raw = sensor_data[left_key]["rgb"][0].cpu().numpy()
        except Exception:
            pass

    if head_rgb_raw is None:
        head_rgb_raw = np.zeros((360, 640, 3), dtype=np.uint8)
    if left_wrist_rgb_raw is None:
        left_wrist_rgb_raw = np.zeros((128, 128, 3), dtype=np.uint8)

    # Right wrist: try dedicated camera, else duplicate left
    right_wrist_rgb_raw = None
    right_key = sensor_names.get("right_wrist_camera")
    if right_key and right_key in sensor_data and right_key != left_key:
        try:
            right_wrist_rgb_raw = sensor_data[right_key]["rgb"][0].cpu().numpy()
        except Exception:
            pass
    if right_wrist_rgb_raw is None:
        right_wrist_rgb_raw = left_wrist_rgb_raw.copy()

    qpos = obs["agent"]["qpos"][0].cpu().numpy()

    # Resize and convert to (C, H, W) float32 [0, 1]
    head_rgb = cv2.resize(head_rgb_raw, (image_size, image_size))
    head_rgb = head_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    left_wrist = cv2.resize(left_wrist_rgb_raw, (image_size, image_size))
    left_wrist = left_wrist.transpose(2, 0, 1).astype(np.float32) / 255.0
    right_wrist = cv2.resize(right_wrist_rgb_raw, (image_size, image_size))
    right_wrist = right_wrist.transpose(2, 0, 1).astype(np.float32) / 255.0

    return {
        "head_rgb": head_rgb,
        "left_wrist_rgb": left_wrist,
        "right_wrist_rgb": right_wrist,
        "head_rgb_raw": head_rgb_raw,
        "left_wrist_rgb_raw": left_wrist_rgb_raw,
        "right_wrist_rgb_raw": right_wrist_rgb_raw,
        "state": {"default": qpos.astype(np.float32)},
    }


# ------------------------------------------------------------------
# 3. Action prediction
# ------------------------------------------------------------------

def predict_action(policy, processor, obs: Dict, task: str,
                   coarse_task: str, action_dim: int,
                   device: str = "cuda") -> np.ndarray:
    """Predict action chunk from observation.

    Returns:
        np.ndarray of shape (chunk_size, action_dim)
    """
    from galaxea_fm.utils.pytorch_utils import dict_apply

    sample = {
        "images": {
            "head_rgb": obs["head_rgb"],
            "left_wrist_rgb": obs["left_wrist_rgb"],
            "right_wrist_rgb": obs["right_wrist_rgb"],
        },
        "state": obs["state"],
        "task": task,
        "coarse_task": coarse_task,
        "state_is_pad": torch.tensor([False]),
        "image_is_pad": torch.tensor([False]),
        "action_is_pad": torch.tensor([False] * 32),
        "idx": torch.tensor(0),
    }
    sample = processor.preprocess(sample)
    batch = dict_apply(sample, lambda x: x.unsqueeze(0).to(device)
                       if isinstance(x, torch.Tensor) else x)
    with torch.no_grad():
        batch = policy.predict_action(batch)
    batch = dict_apply(batch, lambda x: x.cpu() if hasattr(x, "cpu") else x)
    batch = processor.postprocess(batch)
    action = batch["action"]
    if isinstance(action, dict):
        action = action["default"]
    # action shape: (B, chunk_size, action_dim)
    return np.asarray(action).reshape(-1, action_dim)


def vla_action_to_env_action(vla_action: np.ndarray) -> np.ndarray:
    """Convert VLA action to env-compatible action.

    VLA outputs 13-DoF (pd_joint_pos). The env expects the same 13-DoF.
    We zero out head joints (indices 8, 9) to prevent head drift,
    matching eval_policy_client.py behavior.
    """
    action = vla_action.astype(np.float32)
    # Zero out head_pan and head_tilt to prevent head drift during manipulation
    if len(action) > 9:
        action[8] = 0  # head_pan
        action[9] = 0  # head_tilt
    return action


# ------------------------------------------------------------------
# 4. Video saving
# ------------------------------------------------------------------

def save_video(frames: List[np.ndarray], path: str, fps: int = 30):
    """Save list of RGB frames to mp4."""
    import imageio
    writer = imageio.get_writer(path, fps=fps)
    for f in frames:
        writer.append_data(f)
    writer.close()


# ------------------------------------------------------------------
# 5. Environment creation
# ------------------------------------------------------------------

def make_env(scene_dir: str, env_name: Optional[str] = None,
             sim_backend: str = "auto", shader: str = "default",
             robot_uids: str = "ds_fetch_basket"):
    """Create RoboBenchMart gym environment."""
    sys.path.append(RBM_ROOT)
    import gymnasium as gym
    import mani_skill  # noqa: F401 — registers envs

    if env_name is None:
        json_path = os.path.join(scene_dir, "episode_metadata.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
            env_name = meta.get("env_info", {}).get("env_id")
        if env_name is None:
            raise ValueError(
                "Cannot auto-detect env_name. "
                "Provide --env-name or ensure episode_metadata.json exists in scene dir."
            )

    env = gym.make(
        env_name,
        robot_uids=robot_uids,
        config_dir_path=scene_dir,
        num_envs=1,
        control_mode="pd_joint_pos",
        viewer_camera_configs={"shader_pack": shader},
        human_render_camera_configs={"shader_pack": shader},
        sim_backend=sim_backend,
        render_mode="rgb_array",
        enable_shadow=True,
        obs_mode="rgb",
        parallel_in_single_scene=False,
    )
    return env


# ------------------------------------------------------------------
# 6. Single episode runner
# ------------------------------------------------------------------

def run_episode(env, policy, processor,
                coarse_task: str, action_dim: int, max_horizon: int,
                replan_steps: int = 5, save_video_flag: bool = False,
                seed: int = 0) -> Dict:
    """Run one closed-loop episode. Returns dict with success, steps, frames."""
    obs, info = env.reset(seed=seed, options={"reconfigure": True})

    # Get language instruction
    try:
        language_instruction = env.language_instructions[0]
    except (AttributeError, IndexError):
        language_instruction = "pick the item"

    # Auto-detect sensor names from first observation
    sensor_names = get_sensor_names(obs)
    logger.info(f"Detected sensors: {sensor_names}")

    frames_head = [] if save_video_flag else None
    frames_third = [] if save_video_flag else None

    step = 0
    done = False
    truncated = False

    def _to_bool(val):
        """Convert ManiSkill done/truncated (possibly array/tensor) to bool."""
        if isinstance(val, (bool, int, float)):
            return bool(val)
        if isinstance(val, torch.Tensor):
            return bool(val.flatten()[0].item())
        if isinstance(val, np.ndarray):
            return bool(val.flatten()[0])
        return bool(val)

    while step < max_horizon and not _to_bool(done) and not _to_bool(truncated):
        vla_obs = build_vla_obs(obs, sensor_names)

        if save_video_flag:
            frames_head.append(vla_obs["head_rgb_raw"].copy())
            try:
                rendered = env.render()
                if rendered is not None:
                    # ManiSkill render may return (1, H, W, 3) or (H, W, 3)
                    if isinstance(rendered, np.ndarray):
                        if rendered.ndim == 4:
                            rendered = rendered[0]
                    frames_third.append(rendered)
            except Exception:
                pass

        # Predict action chunk: (chunk_size, action_dim)
        action_chunk = predict_action(
            policy, processor, vla_obs,
            task=language_instruction,
            coarse_task=coarse_task,
            action_dim=action_dim,
        )

        # Execute replan_steps actions from the chunk
        for i in range(min(replan_steps, action_chunk.shape[0])):
            env_action = vla_action_to_env_action(action_chunk[i])
            obs, reward, done, truncated, info = env.step(env_action)
            step += 1

            if _to_bool(done) or _to_bool(truncated):
                break

    # Extract success from info
    raw_success = info.get("success", False)
    if isinstance(raw_success, torch.Tensor):
        success = bool(raw_success.flatten()[0].item())
    elif isinstance(raw_success, (list, np.ndarray)):
        success = bool(np.asarray(raw_success).flatten()[0])
    else:
        success = bool(raw_success)

    return {
        "success": success,
        "steps": step,
        "instruction": language_instruction,
        "frames_head": frames_head,
        "frames_third": frames_third,
    }


# ------------------------------------------------------------------
# 7. Main — Hydra entry point
# ------------------------------------------------------------------

@torch.no_grad()
def eval_main(cfg: DictConfig) -> None:
    """Core evaluation logic, receives a fully-resolved Hydra config."""
    from accelerate import Accelerator
    from galaxea_fm.utils.config_resolvers import register_default_resolvers

    # Register custom resolvers (split, sum_shapes, etc.) before resolve
    register_default_resolvers()
    OmegaConf.resolve(cfg)

    # Initialize Accelerator (required by GalaxeaZeroPolicy.__init__)
    gpu_id = int(os.environ.get("LOCAL_RANK", os.environ.get("EVAL_GPU", "0")))
    torch.cuda.set_device(gpu_id)
    accelerator = Accelerator(mixed_precision="bf16")
    _ = accelerator  # keep alive for instantiate

    # Extract eval-specific params
    ckpt_path = cfg.get("ckpt_path", None)
    scene_dir = (cfg.get("eval_scene_dir", None)
                 or os.environ.get("EVAL_SCENE_DIR", ""))
    env_name = (cfg.get("eval_env_name", None)
                or os.environ.get("EVAL_ENV_NAME", None))
    coarse_task = cfg.get("eval_coarse_task", "")
    num_traj = cfg.get("eval_num_traj", 10)
    max_horizon = cfg.get("eval_max_horizon", 300)
    replan_steps = cfg.get("eval_replan_steps", 5)
    start_seed = cfg.get("eval_start_seed", 0)
    sim_backend = cfg.get("eval_sim_backend", "auto")
    shader = cfg.get("eval_shader", "default")
    save_video_flag = cfg.get("eval_save_video", False)
    video_dir = cfg.get("eval_video_dir", None)
    device = f"cuda:{gpu_id}"

    # Determine action_dim from config (should be 13 for Fetch pd_joint_pos)
    try:
        action_dim = OmegaConf.select(cfg, "data.processor.action_output_dim",
                                      default=FETCH_ACTION_DIM)
    except Exception:
        action_dim = FETCH_ACTION_DIM
    logger.info(f"Action dim from config: {action_dim}")

    assert ckpt_path, "Must set ckpt_path (via +ckpt_path=...)"
    assert scene_dir, "Must set eval_scene_dir (via +eval_scene_dir=... or EVAL_SCENE_DIR env)"

    # Load policy
    policy, processor = load_policy(ckpt_path, cfg, device)

    # Create environment
    env = make_env(scene_dir, env_name, sim_backend, shader)

    # Verify action space matches
    env_action_dim = env.action_space.shape[0]
    if action_dim != env_action_dim:
        logger.warning(
            f"Config action_dim={action_dim} != env action_space={env_action_dim}. "
            f"Using env action_space dimension.")
        action_dim = env_action_dim

    # Video output dir
    if not video_dir:
        video_dir = os.path.join(scene_dir, "eval_videos")
    if save_video_flag:
        os.makedirs(video_dir, exist_ok=True)
        logger.info(f"Videos will be saved to {video_dir}")

    # Run episodes
    results = []
    from tqdm import tqdm
    pbar = tqdm(range(num_traj), desc="Evaluating")

    for traj_idx in pbar:
        seed = start_seed + traj_idx

        ep = run_episode(
            env, policy, processor,
            coarse_task=coarse_task,
            action_dim=action_dim,
            max_horizon=max_horizon,
            replan_steps=replan_steps,
            save_video_flag=save_video_flag,
            seed=seed,
        )

        results.append({
            "traj_idx": traj_idx,
            "seed": seed,
            "success": ep["success"],
            "steps": ep["steps"],
            "instruction": ep["instruction"],
        })

        if save_video_flag and ep["frames_head"]:
            tag = f"ep{traj_idx:03d}_seed{seed}"
            save_video(ep["frames_head"],
                       os.path.join(video_dir, f"{tag}_head.mp4"), fps=30)
            if ep["frames_third"]:
                save_video(ep["frames_third"],
                           os.path.join(video_dir, f"{tag}_third.mp4"), fps=30)

        successes_so_far = sum(r["success"] for r in results)
        sr = successes_so_far / len(results) * 100
        pbar.set_postfix({"success_rate": f"{sr:.1f}%", "successes": successes_so_far})

    env.close()

    # Report
    successes = sum(r["success"] for r in results)
    total = len(results)
    avg_steps = np.mean([r["steps"] for r in results])
    success_seeds = [r["seed"] for r in results if r["success"]]

    print("\n" + "=" * 60)
    print(f"  Scene: {scene_dir}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Action dim: {action_dim}")
    print(f"  Episodes: {total}")
    print(f"  Successes: {successes}")
    print(f"  Success Rate: {successes / total * 100:.1f}%")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print(f"  Successful seeds: {success_seeds}")
    print("=" * 60)

    # Save results json
    results_path = os.path.join(video_dir if save_video_flag else scene_dir,
                                "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "scene_dir": scene_dir,
            "ckpt_path": ckpt_path,
            "action_dim": action_dim,
            "num_traj": total,
            "successes": successes,
            "success_rate": successes / total * 100,
            "avg_steps": avg_steps,
            "episodes": results,
        }, f, indent=2)
    logger.info(f"Results saved to {results_path}")


# ------------------------------------------------------------------
# Hydra decorator — same pattern as eval_libero.py
# ------------------------------------------------------------------

import hydra


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    eval_main(cfg)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
