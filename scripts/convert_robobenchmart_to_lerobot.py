"""
Convert RoboBenchMart replayed h5 trajectories to LeRobot v2.1 format
compatible with GalaxeaVLA training.

Usage:
    python scripts/convert_robobenchmart_to_lerobot.py \
        --h5-dir generated_envs/layout1/demos/motionplanning/ \
        --output-dir datasets/supermarket_fetch/ \
        --fps 15

Each replayed h5 file (from `replay_trajectory.py --obs-mode rgbd --save-traj`)
is expected to contain:
  traj_{id}/
    observations/
      head_camera_rgb:  (T, 256, 256, 3) uint8
      fetch_hand_rgb:   (T, 128, 128, 3) uint8
    actions:            (T, 15) float64
    robot_qpos:         (T, 15) float64  (from env state or observations)

Output LeRobot structure:
  {output_dir}/
    meta/
      info.json
      tasks.jsonl
      episodes.jsonl
    data/chunk-000/
      episode_000000.parquet
      ...
    videos/chunk-000/
      observation.images.head_rgb/episode_000000.mp4
      observation.images.left_wrist_rgb/episode_000000.mp4
      observation.images.right_wrist_rgb/episode_000000.mp4
"""
import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Try to import video encoding
try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

# Try LeRobot import
try:
    from galaxea_fm.data.lerobot.datasets.utils import (
        write_json, write_info, DEFAULT_FEATURES,
    )
    HAS_LEROBOT_UTILS = True
except ImportError:
    HAS_LEROBOT_UTILS = False


# ============================================================
# Camera mapping: RoboBenchMart → G0/LeRobot format
# ============================================================
CAMERA_KEY_MAP = {
    "head_camera": "head_rgb",
    "head_camera_rgb": "head_rgb",
    "fetch_hand": "left_wrist_rgb",
    "fetch_hand_rgb": "left_wrist_rgb",
}

# We duplicate left wrist to right wrist (Fetch only has 1 wrist camera)
DUPLICATE_WRIST = True  # left_wrist_rgb → right_wrist_rgb

# ============================================================
# State extraction
# ============================================================
def extract_robot_qpos(traj_group):
    """Extract robot qpos from trajectory group. Tries multiple keys."""
    # Try direct key
    if "robot_qpos" in traj_group:
        return np.array(traj_group["robot_qpos"])

    # Try env_states
    if "env_states" in traj_group:
        env_states = traj_group["env_states"]
        # env_states is typically a dict of arrays
        # Try to find agent qpos
        for key in env_states.keys():
            if "qpos" in key.lower() or "agent" in key.lower():
                return np.array(env_states[key])

    # Try observations
    if "observations" in traj_group:
        obs = traj_group["observations"]
        for key in obs.keys():
            if "qpos" in key.lower() or "state" in key.lower():
                val = np.array(obs[key])
                if val.ndim == 2:
                    return val

    raise KeyError(f"Cannot find robot qpos in trajectory. Available keys: {list(traj_group.keys())}")


def extract_actions(traj_group):
    """Extract actions from trajectory group."""
    if "actions" in traj_group:
        return np.array(traj_group["actions"])
    raise KeyError(f"Cannot find actions in trajectory. Available keys: {list(traj_group.keys())}")


def extract_images(traj_group, camera_key: str):
    """Extract images for a specific camera from trajectory group."""
    obs = traj_group["observations"]
    # Try various naming conventions
    for suffix in ["_rgb", "_color", ""]:
        key = f"{camera_key}{suffix}"
        if key in obs:
            return np.array(obs[key])
    raise KeyError(f"Cannot find images for camera '{camera_key}' in observations. "
                   f"Available keys: {list(obs.keys())}")


# ============================================================
# Video encoding
# ============================================================
def encode_frames_to_mp4(frames: np.ndarray, output_path: Path, fps: int = 15):
    """Encode numpy frames (T, H, W, 3) uint8 to mp4 video."""
    if not HAS_AV:
        raise ImportError("av (PyAV) is required for video encoding. Install: pip install av")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, C = frames.shape
    assert C == 3, f"Expected 3 channels, got {C}"

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "medium"}

    for t in range(T):
        frame = av.VideoFrame.from_ndarray(frames[t], format="rgb24")
        frame.pts = t
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush
    for packet in stream.encode():
        container.mux(packet)
    container.close()


# ============================================================
# LeRobot metadata writers
# ============================================================
def write_lerobot_metadata(output_dir: Path, fps: int, total_frames: int,
                           total_episodes: int, feature_keys: list):
    """Write meta/info.json, meta/tasks.jsonl, meta/episodes.jsonl."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "fetch",
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float64",
                "shape": [32, 15],
                "names": ["base_x", "base_y", "base_yaw", "torso_lift",
                          "shoulder_pan", "shoulder_lift", "upperarm_roll",
                          "elbow_flex", "forearm_roll", "wrist_flex", "wrist_roll",
                          "head_pan", "head_tilt", "r_gripper", "l_gripper"],
            },
            "observation.state": {
                "dtype": "float64",
                "shape": [15],
                "names": ["base_x", "base_y", "base_yaw", "torso_lift",
                          "shoulder_pan", "shoulder_lift", "upperarm_roll",
                          "elbow_flex", "forearm_roll", "wrist_flex", "wrist_roll",
                          "head_pan", "head_tilt", "r_gripper", "l_gripper"],
            },
        },
        "splits": {"train": f"0:{total_episodes}"},
    }
    # Add image features
    for camera_lerobot_key, (H, W) in feature_keys:
        info["features"][f"observation.images.{camera_lerobot_key}"] = {
            "dtype": "video",
            "shape": [H, W, 3],
            "names": ["height", "width", "channels"],
        }

    write_json(info, meta_dir / "info.json")

    # tasks.jsonl (single task for now; can be extended per trajectory)
    tasks = [{"task_index": 0, "task": "robobenchmart_task"}]
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def build_parquet_table(episode_data: dict) -> pa.Table:
    """Build a PyArrow table from episode data dict."""
    columns = {}
    for key, value in episode_data.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                columns[key] = pa.array(value.tolist())
            elif value.ndim == 2:
                # Store 2D arrays as list of lists
                columns[key] = pa.array([row.tolist() for row in value])
            else:
                columns[key] = pa.array(value.flatten().tolist())
        elif isinstance(value, list):
            columns[key] = pa.array(value)
        else:
            columns[key] = pa.array([value])
    return pa.table(columns)


# ============================================================
# Main conversion
# ============================================================
def convert_trajectories(
    h5_dir: Path,
    output_dir: Path,
    fps: int = 15,
    task_description: str = "pick the item and put to the basket",
):
    """Convert all replayed h5 files in h5_dir to LeRobot format."""

    # Find all replayed h5 files (those with rgbd in name)
    h5_files = sorted(h5_dir.glob("*.rgbd.*.h5"))
    if not h5_files:
        h5_files = sorted(h5_dir.glob("*.h5"))

    if not h5_files:
        raise FileNotFoundError(f"No h5 files found in {h5_dir}")

    print(f"Found {len(h5_files)} h5 files")

    output_dir = Path(output_dir)
    data_dir = output_dir / "data" / "chunk-000"
    video_dir = output_dir / "videos" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    episode_index = 0
    episode_meta = []

    # Determine LeRobot camera keys
    lerobot_camera_keys = ["head_rgb", "left_wrist_rgb"]
    if DUPLICATE_WRIST:
        lerobot_camera_keys.append("right_wrist_rgb")

    for video_key in lerobot_camera_keys:
        (video_dir / f"observation.images.{video_key}").mkdir(parents=True, exist_ok=True)

    for h5_path in tqdm(h5_files, desc="Converting h5 files"):
        h5_name = h5_path.stem

        with h5py.File(h5_path, "r") as f:
            # Count trajectories
            traj_ids = sorted([k for k in f.keys() if k.startswith("traj_")])

            for traj_id in tqdm(traj_ids, desc=f"  {h5_name}", leave=False):
                traj = f[traj_id]

                try:
                    actions = extract_actions(traj)
                    qpos = extract_robot_qpos(traj)

                    # Extract images from both cameras
                    head_frames = extract_images(traj, "head_camera")
                    wrist_frames = extract_images(traj, "fetch_hand")
                except KeyError as e:
                    print(f"  Skipping {traj_id}: {e}")
                    continue

                T = len(actions)
                assert T == len(qpos) == len(head_frames) == len(wrist_frames), \
                    f"Length mismatch: actions={len(actions)}, qpos={len(qpos)}, " \
                    f"head={len(head_frames)}, wrist={len(wrist_frames)}"

                if T < 33:  # Need at least 32+1 frames for sliding window
                    print(f"  Skipping {traj_id}: too short ({T} frames)")
                    continue

                # ============================================
                # Encode videos
                # ============================================
                ep_str = f"episode_{episode_index:06d}"

                video_paths = {}
                for cam_rbm, cam_lerobot in [("head_camera", "head_rgb"),
                                              ("fetch_hand", "left_wrist_rgb")]:
                    frames = head_frames if cam_rbm == "head_camera" else wrist_frames
                    vpath = video_dir / f"observation.images.{cam_lerobot}" / f"{ep_str}.mp4"
                    encode_frames_to_mp4(frames, vpath, fps)
                    video_paths[cam_lerobot] = str(vpath)

                if DUPLICATE_WRIST:
                    # Copy left wrist video as right wrist
                    src = video_paths["left_wrist_rgb"]
                    dst = video_dir / f"observation.images.right_wrist_rgb" / f"{ep_str}.mp4"
                    shutil.copy2(src, dst)

                # ============================================
                # Build parquet data
                # ============================================
                parquet_data = {
                    "observation.state": qpos.astype(np.float64),       # (T, 15)
                    "action": actions.astype(np.float64),                # (T, 15)
                    "episode_index": np.full(T, episode_index, dtype=np.int64),
                    "frame_index": np.arange(T, dtype=np.int64),
                    "index": np.arange(total_frames, total_frames + T, dtype=np.int64),
                    "timestamp": np.arange(T, dtype=np.float32) / fps,
                    "task_index": np.zeros(T, dtype=np.int64),          # All same task
                    "coarse_task_index": np.zeros(T, dtype=np.int64),
                    "quality_index": np.full(T, 100, dtype=np.int64),
                    "coarse_quality_index": np.full(T, 100, dtype=np.int64),
                }

                table = build_parquet_table(parquet_data)
                pq.write_table(table, data_dir / f"{ep_str}.parquet")

                # ============================================
                # Episode metadata
                # ============================================
                episode_meta.append({
                    "episode_index": episode_index,
                    "tasks": [task_description],
                    "length": int(T),
                })

                total_frames += T
                episode_index += 1

    # ============================================
    # Write metadata
    # ============================================
    print(f"\nTotal: {episode_index} episodes, {total_frames} frames")

    write_lerobot_metadata(
        output_dir=output_dir,
        fps=fps,
        total_frames=total_frames,
        total_episodes=episode_index,
        feature_keys=[
            ("head_rgb", (256, 256)),
            ("left_wrist_rgb", (128, 128)),
            ("right_wrist_rgb", (128, 128)),
        ],
    )

    # episodes.jsonl
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for ep in episode_meta:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # tasks.jsonl
    tasks = [{"task_index": 0, "task": task_description}]
    with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Conversion complete: {output_dir}")
    return episode_index, total_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert RoboBenchMart replayed h5 to LeRobot format for GalaxeaVLA")
    parser.add_argument("--h5-dir", type=str, required=True,
                        help="Directory containing replayed h5 files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for LeRobot dataset")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second")
    parser.add_argument("--task", type=str, default="pick the item and put to the basket",
                        help="Task description for all episodes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_trajectories(
        h5_dir=Path(args.h5_dir),
        output_dir=Path(args.output_dir),
        fps=args.fps,
        task_description=args.task,
    )
