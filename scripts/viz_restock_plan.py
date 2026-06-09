"""把 plan.json 的导航路口画到俯视图上: 标点 + 红线, 取货前/后切成两张图。

用法:
  P=/home/lh/software/miniconda3/envs/robort_mart/bin/python
  cd /home/lh/VLA/GalaxeaVLA-main
  $P scripts/viz_restock_plan.py   # 读 restock_sim_plan/plan.json + clean_topdown.png
"""
import os
import sys
import json
import argparse

RBM_ROOT = os.environ.get("RBM_ROOT", "/home/lh/VLA/RoboBenchMart-main")
sys.path.insert(0, RBM_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def nav_points(plan):
    """按出现顺序提取 navigate_to 的世界坐标, 并按第一个操作动作切成两段。"""
    legs, cur = [[]], []
    started_manip = False
    op = {"pick_to_basket", "restock_basket_to_shelf", "pick_from_floor"}
    for s in plan:
        if s.get("type") == "navigate_to":
            try:
                x, y = [float(v) for v in str(s["target"]).split(",")]
            except (ValueError, KeyError):
                continue
            cur.append([x, y])
        elif s.get("type") in op and not started_manip:
            # 第一次操作 = 到达仓库取货, 切段
            started_manip = True
            legs[0] = cur
            cur = [cur[-1]] if cur else []
    legs.append(cur)
    return legs[0], legs[1]


def draw_leg(bg_png, scene_size, pts, out_path, title):
    """在干净俯视图上画一段路径: 路口点 + 红线连接。pts 为世界坐标列表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    sx, sy = scene_size
    fig, ax = plt.subplots(figsize=(sx, sy))
    if bg_png and os.path.exists(bg_png):
        ax.imshow(mpimg.imread(bg_png), extent=[0, sx, 0, sy], origin="upper")
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", color="red", lw=2.5, zorder=3)
        ax.plot(xs, ys, "o", color="red", ms=10, zorder=4)
        for i, (x, y) in enumerate(pts):
            ax.annotate(str(i + 1), (x, y), color="white", fontsize=9,
                        ha="center", va="center", zorder=5,
                        fontweight="bold")
    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)
    ax.set_title(title)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {title} ({len(pts)} pts) -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="restock_sim_plan",
                   help="含 plan.json + clean_topdown.png 的目录")
    p.add_argument("--scene-x", type=float, default=16.0)
    p.add_argument("--scene-y", type=float, default=10.0)
    args = p.parse_args()

    d = os.path.abspath(args.dir)
    plan = json.load(open(os.path.join(d, "plan.json")))["plan"]
    bg = os.path.join(d, "topdown.png")
    scene = [args.scene_x, args.scene_y]

    leg1, leg2 = nav_points(plan)
    draw_leg(bg, scene, leg1, os.path.join(d, "leg1_rest_to_warehouse.png"),
             "Leg 1: rest area -> warehouse shelf")
    draw_leg(bg, scene, leg2, os.path.join(d, "leg2_warehouse_to_shelf.png"),
             "Leg 2: warehouse -> commercial shelf -> back")


if __name__ == "__main__":
    main()
