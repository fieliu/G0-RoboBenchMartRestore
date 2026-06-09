#!/usr/bin/env python3
"""Closed-loop evaluation on RoboBenchMart environments.

Directly loads a LoRA-finetuned G0Plus policy (no websocket / port needed),
runs N episodes, and reports per-scene + aggregate success rates.

Usage:
    python scripts/eval_robobenchmart.py \
        --scene-dir /path/to/RoboBenchMart-main/demo_envs/pick_to_basket \
        --env-name PickToBasketContNiveaEnv \
        --ckpt-path ./ckpts/fetch_lora_best/best_model.pt \
        --num-traj 10 --save-video

If --env-name is omitted, it is auto-detected from the scene's json metadata.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EvalRoboBenchMart")

RBM_ROOT = os.environ.get("RBM_ROOT", "/home/lh/VLA/RoboBenchMart-main")


# ------------------------------------------------------------------
# 1. Model loading (same as VLAExecutor in deploy_supermarket.py)
# ------------------------------------------------------------------

def load_policy(ckpt_path: str,
                cfg_path: str = "configs/task/robobenchmart/fetch_lora_finetune.yaml",
                device: str = "cuda"):
    """Load LoRA-finetuned G0Plus policy + processor."""
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from galaxea_fm.utils.config_resolvers import register_default_resolvers
    from galaxea_fm.utils.load_pretrained_resumed import load_checkpoint_for_eval

    register_default_resolvers()
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)

    policy = instantiate(cfg.model.model_arch)
    policy, stats = load_checkpoint_for_eval(ckpt_path, policy, device="cpu")
    policy = policy.to(device).eval()

    processor = instantiate(cfg.data.processor)
    processor.set_normalizer_from_stats(stats)
    processor.eval()

    logger.info(f"Loaded policy from {ckpt_path}")
    return policy, processor, cfg


# ------------------------------------------------------------------
# 2. Observation construction (from deploy_supermarket.py SimEnv.get_obs)
# ------------------------------------------------------------------

def get_sensor_names(env):
    """Auto-detect sensor camera names from ManiSkill scene."""
    try:
        sensor_data = env.scene.get_sensor_data()
        all_keys = list(sensor_data.keys()) if isinstance(sensor_data, dict) else []
    except Exception:
        all_keys = []

    names = {"head_camera": None, "left_wrist_camera": None, "right_wrist_camera": None}
    for k in all_keys:
        kl = k.lower()
        if "head" in kl or "agent" in kl or "top" in kl or "ceiling" in kl:
            names["head_camera"] = k
        elif "left" in kl or "wrist" in kl or "hand" in kl:
            names["left_wrist_camera"] = k
            names["right_wrist_camera"] = k  # Fetch has one gripper
    return names


def build_vla_obs(env, sensor_names: dict) -> Dict:
    """Build observation dict for VLA policy from ManiSkill env."""
    import torch
    import cv2

    sensor_data = env.scene.get_sensor_data()

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
    right_wrist_rgb_raw = left_wrist_rgb_raw.copy()

    qpos = env.agent.get_qpos()[0].cpu().numpy()

    head_rgb = cv2.resize(head_rgb_raw, (224, 224)).transpose(2, 0, 1).astype(np.float32) / 255.0
    left_wrist = cv2.resize(left_wrist_rgb_raw, (224, 224)).transpose(2, 0, 1).astype(np.float32) / 255.0
    right_wrist = left_wrist.copy()

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
# 3. Action prediction (same as VLAExecutor.act)
# ------------------------------------------------------------------

def predict_action(policy, processor, obs: Dict, task: str,
                   coarse_task: str, device: str = "cuda") -> np.ndarray:
    """Predict 15-DoF action from observation."""
    import torch
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
    return np.asarray(action).reshape(-1, 15)


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
             sim_backend: str = "auto", shader: str = "default"):
    """Create RoboBenchMart gym environment."""
    sys.path.append(RBM_ROOT)
    import gymnasium as gym
    import mani_skill  # noqa: F401 — registers envs

    # Try to auto-detect env_name from scene metadata
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

def run_episode(env, policy, processor, sensor_names: dict,
                coarse_task: str, max_horizon: int,
                replan_steps: int = 5, save_video_flag: bool = False,
                seed: int = 0) -> Dict:
    """Run one closed-loop episode. Returns dict with success, steps, frames."""
    obs_ms, info = env.reset(seed=seed, options={"reconfigure": True})
    language_instruction = env.language_instructions[0]

    sensor_names = sensor_names or get_sensor_names(env)

    frames_head = [] if save_video_flag else None
    frames_third = [] if save_video_flag else None

    step = 0
    success = False

    while step < max_horizon:
        # Build VLA observation
        vla_obs = build_vla_obs(env, sensor_names)

        # Record frames before action
        if save_video_flag:
            frames_head.append(vla_obs["head_rgb_raw"].copy())
            try:
                rendered = env.render()
                if rendered is not None:
                    frames_third.append(rendered)
            except Exception:
                pass

        # Predict action
        action_chunk = predict_action(
            policy, processor, vla_obs,
            task=language_instruction,
            coarse_task=coarse_task,
        )

        # Execute replan_steps actions from the chunk
        for i in range(min(replan_steps, action_chunk.shape[0])):
            action = action_chunk[i].astype(np.float32)
            # Zero out unused gripper joints (indices 8, 9) — same as eval_policy_client
            action[8] = 0
            action[9] = 0
            obs_ms, reward, done, truncated, info = env.step(action)
            step += 1

            if done or truncated:
                break

        if done or truncated:
            break

    success = bool(info.get("success", [False])[0]
                   if isinstance(info.get("success"), (list, np.ndarray))
                   else info.get("success", False))

    return {
        "success": success,
        "steps": step,
        "instruction": language_instruction,
        "frames_head": frames_head,
        "frames_third": frames_third,
    }


# ------------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Closed-loop eval on RoboBenchMart")
    parser.add_argument("--scene-dir", type=str, required=True,
                        help="Path to scene directory (e.g. demo_envs/pick_to_basket)")
    parser.add_argument("--env-name", type=str, default=None,
                        help="Gym env id. Auto-detected from scene metadata if omitted.")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to LoRA checkpoint (e.g. best_model.pt)")
    parser.add_argument("--cfg-path", type=str,
                        default="configs/task/robobenchmart/fetch_lora_finetune.yaml",
                        help="Training config used for the checkpoint")
    parser.add_argument("--coarse-task", type=str, default="",
                        help="High-level task description for coarse_task channel. "
                             "If empty, uses the env's language_instruction.")
    parser.add_argument("-n", "--num-traj", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-horizon", type=int, default=300,
                        help="Max steps per episode")
    parser.add_argument("--replan-steps", type=int, default=5,
                        help="How many actions to execute per model inference")
    parser.add_argument("--start-seed", type=int, default=0,
                        help="Starting seed for episode randomization")
    parser.add_argument("--sim-backend", type=str, default="auto",
                        choices=["auto", "cpu", "gpu"])
    parser.add_argument("--shader", type=str, default="default",
                        help="Render shader: default (fast), rt, rt-fast")
    parser.add_argument("--save-video", action="store_true",
                        help="Save head-camera + third-person videos per episode")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory to save videos (default: <scene-dir>/eval_videos/)")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load policy
    policy, processor, cfg = load_policy(args.ckpt_path, args.cfg_path, args.device)

    # Create environment
    env = make_env(args.scene_dir, args.env_name, args.sim_backend, args.shader)
    sensor_names = get_sensor_names(env)
    logger.info(f"Sensor names: {sensor_names}")

    # Video output dir
    video_dir = args.video_dir or os.path.join(args.scene_dir, "eval_videos")
    if args.save_video:
        os.makedirs(video_dir, exist_ok=True)
        logger.info(f"Videos will be saved to {video_dir}")

    # Run episodes
    results = []
    pbar = tqdm(range(args.num_traj), desc="Evaluating")

    for traj_idx in pbar:
        seed = args.start_seed + traj_idx
        coarse_task = args.coarse_task or ""

        ep = run_episode(
            env, policy, processor, sensor_names,
            coarse_task=coarse_task,
            max_horizon=args.max_horizon,
            replan_steps=args.replan_steps,
            save_video_flag=args.save_video,
            seed=seed,
        )

        results.append({
            "traj_idx": traj_idx,
            "seed": seed,
            "success": ep["success"],
            "steps": ep["steps"],
            "instruction": ep["instruction"],
        })

        # Save videos
        if args.save_video and ep["frames_head"]:
            tag = f"ep{traj_idx:03d}_seed{seed}"
            save_video(ep["frames_head"],
                       os.path.join(video_dir, f"{tag}_head.mp4"), fps=30)
            if ep["frames_third"]:
                save_video(ep["frames_third"],
                           os.path.join(video_dir, f"{tag}_third.mp4"), fps=30)

        # Update progress bar
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
    print(f"  Scene: {args.scene_dir}")
    print(f"  Checkpoint: {args.ckpt_path}")
    print(f"  Episodes: {total}")
    print(f"  Successes: {successes}")
    print(f"  Success Rate: {successes / total * 100:.1f}%")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print(f"  Successful seeds: {success_seeds}")
    print("=" * 60)

    # Save results json
    results_path = os.path.join(video_dir if args.save_video else args.scene_dir,
                                "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "scene_dir": args.scene_dir,
            "ckpt_path": args.ckpt_path,
            "num_traj": total,
            "successes": successes,
            "success_rate": successes / total * 100,
            "avg_steps": avg_steps,
            "episodes": results,
        }, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
