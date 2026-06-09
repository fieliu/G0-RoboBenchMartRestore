"""在 sim 中启动 RestockFlow 补货环境(机器人位于角落休息区)。

这是给 VLA 模型做仿真测试的入口: 真起 sim、reset、渲染, 并预留 env.step(action)
接口。不含运动规划/求解器 —— 动作由你的 VLA 模型给出。

用法:
  P=/home/lh/software/miniconda3/envs/robort_mart/bin/python
  $P launch_restock_sim.py --env RestockFlowContNiveaEnv \
     --scene-dir generated_envs/restock_scene --out-dir /tmp/restock_sim
"""
import os
import sys
import argparse
import numpy as np

RBM_ROOT = os.environ.get("RBM_ROOT", "/home/lh/VLA/RoboBenchMart-main")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="RestockFlowContNiveaEnv")
    p.add_argument("--scene-dir", default="generated_envs/restock_scene")
    p.add_argument("--robot-uids", default="ds_fetch_basket")
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--out-dir", default="/tmp/restock_sim")
    p.add_argument("--steps", type=int, default=0,
                   help=">0 时用随机动作推进若干步, 验证 step 闭环")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.out_dir = os.path.abspath(args.out_dir)
    sys.path.insert(0, RBM_ROOT)
    # hydra 按 cwd 解析 config_dir_path, 场景在 RBM 根目录下 -> chdir 过去
    scene_dir = args.scene_dir if os.path.isabs(args.scene_dir) \
        else os.path.join(RBM_ROOT, args.scene_dir)
    os.chdir(RBM_ROOT)
    import gymnasium as gym
    import dsynth.envs, dsynth.robots  # noqa: F401  注册环境
    from PIL import Image

    env = gym.make(
        args.env, robot_uids=args.robot_uids,
        config_dir_path=scene_dir, num_envs=1,
        control_mode="pd_joint_pos", render_mode="rgb_array",
        obs_mode="rgbd", enable_shadow=False,
        parallel_in_single_scene=False,
        sim_backend=args.backend, render_backend=args.backend,
    )
    obs, info = env.reset(options={"reconfigure": True})
    u = env.unwrapped

    rpos = u.agent.base_link.pose.p[0].cpu().numpy()
    print(f"[sim] env={args.env}")
    print(f"[sim] robot at rest area: ({rpos[0]:.2f}, {rpos[1]:.2f})")
    print(f"[sim] instruction: {u.language_instructions[0]}")
    raw = u.actors.get("fixtures", {}).get("shelves", {})
    n_active = sum(1 for k in raw if "active" in k and "inactive" not in k)
    print(f"[sim] active shelves: {n_active}, products: {len(u.actors.get('products', {}))}")

    img = np.asarray(env.render())[0]
    out = os.path.join(args.out_dir, "rest_area.png")
    Image.fromarray(img).save(out)
    print(f"[sim] render saved -> {out}")

    for i in range(args.steps):
        # 此处替换为你的 VLA 模型输出: action = policy(obs)
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()
    if args.steps:
        print(f"[sim] stepped {args.steps} steps OK (random action placeholder)")

    print("[sim] ready. 在脚本里把 env.step(action) 的 action 换成你的 VLA 输出即可。")


if __name__ == "__main__":
    main()
