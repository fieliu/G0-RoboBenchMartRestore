"""
VLM Planning-Only Test (RoboBenchMart scene -> VLM inputs -> plan validation)
=============================================================================
只验证 VLM 规划是否正确，不跑运动规划、不执行任何动作。

依赖方向(与 deploy_supermarket.py 的 RealEnvObsProvider 一致):
  GalaxeaVLA(本项目, 大脑) ──import──> RoboBenchMart(环境, 只借 StoreMapProvider/gym env)
  本脚本与 deploy_supermarket.py 同目录, 直接复用其 VLMPlanner。

流程:
  1. gym.make 一个 demo 场景 -> reset
  2. StoreMapProvider 抽取 VLM 规划所需的全部输入:
       - 俯视图 (topdown.png)
       - 物品清单 / 坐标表 / 机器人位姿 (map_block.txt)
  3. (可选) 调 VLM 规划器，把 文本地图 + 俯视图 + 指令 喂进去，拿回子任务序列
  4. 校验规划: 类型合法 / 坐标取自已知点(不瞎编) / 目标物品在清单里 / 序列合理
  5. 落盘 plan.json + 打印 PASS/FAIL

用法 (在 robort_mart 环境里, 从本脚本所在目录运行):
  P=/home/lh/software/miniconda3/envs/robort_mart/bin/python
  cd /home/lh/VLA/GalaxeaVLA-main

  # A) 只抽取 VLM 输入(离线, 不需要任何 VLM, 先看俯视图和文本地图对不对)
  $P scripts/test_vlm_planning.py \
      --scene-dir /home/lh/VLA/RoboBenchMart-main/demo_envs/pick_to_basket \
      --env-name PickToBasketContNiveaEnv --out-dir vlm_plan_test

  # B) 抽取 + 调 VLM 规划 + 校验 (需先 pip install anthropic 或 openai)
  $P scripts/test_vlm_planning.py \
      --scene-dir /home/lh/VLA/RoboBenchMart-main/demo_envs/pick_to_basket \
      --env-name PickToBasketContNiveaEnv --out-dir vlm_plan_test \
      --command "把 Fanta 放进篮子" --target-product Fanta \
      --call-vlm --vlm-provider anthropic
"""
import os
import sys
import json
import argparse

# 本脚本与 deploy_supermarket.py 同目录 -> 直接 import 同级模块。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 单向依赖: 仅把 RoboBenchMart 加进 path 以 import dsynth(环境侧)。
RBM_ROOT = os.environ.get("RBM_ROOT", "/home/lh/VLA/RoboBenchMart-main")
sys.path.insert(0, RBM_ROOT)

import numpy as np


# ============================================================
# 1. 起场景 + 抽取 VLM 规划输入
# ============================================================
def build_scene_and_extract(scene_dir, env_name, robot_uids, sim_backend, out_dir):
    """gym.make -> reset -> 复用 RoboBenchMart 自带的 generate_topdown_map 产出全场景俯视图。

    俯视图与全部货架坐标直接调用 RoboBenchMart 的 scripts/generate_topdown_map.py
    (matplotlib 画的 1:1 全场景图, 每个货架带 ID+坐标), 不自己渲染。
    只额外用 sim 修正该脚本拿不到的真值: active 商品货架真实位置(layout 里是占位 0,0)
    与机器人起始位姿。返回 (env, map_block, facts, topdown_png_path)。
    """
    import gymnasium as gym
    import dsynth.envs  # noqa: F401  触发环境注册(装饰器在 import 时执行)
    import dsynth.robots  # noqa: F401  触发机器人注册
    sys.path.insert(0, os.path.join(RBM_ROOT, "scripts"))
    from generate_topdown_map import generate_map

    os.makedirs(out_dir, exist_ok=True)  # generate_map 落盘前必须存在

    # RoboBenchMart 用相对路径加载资产(assets/...), 必须在其根目录下运行。
    os.chdir(RBM_ROOT)

    print(f"[scene] gym.make({env_name}, scene_dir={scene_dir}) ...")
    env = gym.make(
        env_name, robot_uids=robot_uids, config_dir_path=scene_dir, num_envs=1,
        control_mode="pd_joint_pos", render_mode="rgb_array", obs_mode="rgbd",
        enable_shadow=False, parallel_in_single_scene=False,
        sim_backend=sim_backend,
        render_backend="cpu" if sim_backend == "cpu" else "gpu",
    )
    env.reset(options={"reconfigure": True})
    print("[scene] reset done")

    robot, active = _sim_truth(env)

    # 全场景俯视图 + 全部货架坐标: 直接用 RoboBenchMart 自带脚本(不自己画)。
    # 传入 sim 真值修正其写死项: 机器人位置、active 货架占位坐标、标题。
    png_path = os.path.join(out_dir, "topdown.png")
    active_pos = active["center"] if active else None
    _, coords_json = generate_map(
        scene_dir, output_path=png_path, robot_angle=robot["yaw_deg"],
        robot_pos=robot["pos"], active_pos=active_pos,
        title=f"{env_name} — Top-Down Map",
        show_zones=False, show_coord_table=False)

    # facts 复用 layout 的丰富构建(仓库/商业分区 + 每架 approach + 商品名),
    # 再用 sim 真值覆盖: 机器人休息区起点 + 商业占位货架真实坐标/接近点。
    facts = _build_facts_from_layout(coords_json, rest_area=robot["pos"])
    facts["robot"] = robot
    facts["waypoints"]["rest_area"] = robot["pos"]
    if active is not None:
        for sid, s in facts["shelves"].items():
            c = s["center"]
            if s.get("role") == "commercial" and abs(c[0]) < 0.01 and abs(c[1]) < 0.01:
                s["center"] = active["center"]
                s["approach"] = active["approach"]
                s["face_deg"] = active["face_deg"]
                facts["waypoints"][f"{sid}_approach"] = active["approach"]
    map_block = _build_map_block(facts)

    print(f"[map] {len(facts['shelves'])} shelves, robot at {robot['pos']}, "
          f"scene {facts['scene_size'][0]:.1f}x{facts['scene_size'][1]:.1f}m")
    return env, map_block, facts, png_path


def _sim_truth(env):
    """从 sim 取 generate_topdown_map 拿不到的真值: 机器人位姿 + active 货架真实位置/物品层。"""
    u = env.unwrapped

    mat = u.agent.base_link.pose.to_transformation_matrix()[0].cpu().numpy()
    rpos = u.agent.base_link.pose.p[0].cpu().numpy()
    ryaw = float(np.degrees(np.arctan2(mat[1, 0], mat[0, 0])))
    robot = {"pos": [round(float(rpos[0]), 2), round(float(rpos[1]), 2)],
             "yaw_deg": round(ryaw, 0)}

    # active 商品货架真实位姿 + 接近点。单货架场景(pick_to_basket)抓唯一 active 货架;
    # 多货架场景(RestockFlow)优先抓 commercial 货架——它在 layout 里是占位 (0,0),
    # 必须用 sim 真值修正; 仓库货架在 layout 里已有真实坐标, 无需 sim 修。
    active = None
    raw = u.actors.get("fixtures", {}).get("shelves", {})
    active_candidates = [(n, a) for n, a in raw.items()
                         if "active" in n and "inactive" not in n]
    # 多货架时优先 commercial; 否则取第一个 active
    commercial = [(n, a) for n, a in active_candidates if "commercial" in n.lower()]
    pick = commercial[0] if commercial else (active_candidates[0] if active_candidates else None)
    if pick is not None:
        actor_name, actor = pick
        m = actor.pose.sp.to_transformation_matrix()
        pos = np.asarray(actor.pose.sp.p)
        facing = m[:3, 1]
        approach = pos[:2] - 1.4 * facing[:2]
        active = {
            "center": [round(float(pos[0]), 2), round(float(pos[1]), 2)],
            "approach": [round(float(approach[0]), 2), round(float(approach[1]), 2)],
            "face_deg": round(float(np.degrees(np.arctan2(facing[1], facing[0]))), 0),
            "layers": {},
        }

    # products_df: 按层(board)聚合物品名
    df = getattr(u, "products_df", None)
    if active is not None and df is not None:
        layers = {}
        for _, row in df.iterrows():
            b = str(row.get("board_idxs", "?"))
            layers.setdefault(b, set()).add(str(row.get("product_name", "unknown")))
        active["layers"] = {b: sorted(v) for b, v in sorted(layers.items())}
    return robot, active


def _build_facts(coords_json, robot, active):
    """合并 generate_topdown_map 的全货架坐标 + sim 真值, 构建规划校验用 facts。"""
    cd = json.load(open(coords_json))
    scene_size = [cd["room"]["size_x"], cd["room"]["size_y"]]

    shelves = {}
    for s in cd["commercial_shelves"] + cd["warehouse_shelves"]:
        center = [s["center"]["x"], s["center"]["y"]]
        entry = {"center": center, "products": s.get("products", []),
                 "n_products": s.get("product_count", 0)}
        # 只有 layout 里坐标≈(0,0) 的占位货架(商业 active 货架)才用 sim 真值修正。
        # 仓库货架在 layout 里已有真实坐标, 不能覆盖。
        is_placeholder = abs(center[0]) < 0.01 and abs(center[1]) < 0.01
        if active is not None and is_placeholder:
            entry["center"] = active["center"]
            entry["approach"] = active["approach"]
            entry["face_deg"] = active["face_deg"]
            entry["layers"] = active["layers"]
        shelves[s["id"]] = entry

    # nav 合法目标: 各货架接近点(有的话) + 机器人起点
    waypoints = {"start": robot["pos"]}
    for sid, s in shelves.items():
        if "approach" in s:
            waypoints[f"{sid}_approach"] = s["approach"]
    return {"scene_size": scene_size, "robot": robot,
            "waypoints": waypoints, "shelves": shelves}


def _approach_point(center, orientation, scene_size):
    """从货架中心+朝向近似算接近点(站位)+ 面向角。不依赖 sim, 供 layout 路径用。

    机器人站在货架靠走廊一侧 1.2m 处, 面朝货架。几何是近似的:
    规划测试只要求坐标取自这份候选清单, 不要求亚米级精度。
    """
    cx, cy = center
    sx, sy = scene_size
    off = 1.2
    if orientation == "horizontal":   # 长轴沿 x, 货架面朝 +/-y
        dy = off if cy < sy / 2 else -off
        return [round(cx, 2), round(cy + dy, 2)], round(-90 if dy > 0 else 90, 0)
    else:                              # 长轴沿 y, 货架面朝 +/-x
        dx = off if cx < sx / 2 else -off
        return [round(cx + dx, 2), round(cy, 2)], round(180 if dx > 0 else 0, 0)


def _build_facts_from_layout(coords_json, rest_area=None):
    """不起 sim, 直接从 generate_topdown_map 的 coords JSON 构建 restock 规划 facts。

    适配多货架 restock 场景: 仓库货架(warehouse, 有库存=取货源) +
    商业货架(commercial, 待补货目标)。机器人起点用休息区(rest_area)。
    """
    cd = json.load(open(coords_json))
    scene_size = [cd["room"]["size_x"], cd["room"]["size_y"]]

    shelves = {}
    for role, key in (("warehouse", "warehouse_shelves"),
                      ("commercial", "commercial_shelves")):
        for s in cd.get(key, []):
            center = [s["center"]["x"], s["center"]["y"]]
            ori = s.get("orientation", "horizontal")
            ap, face = _approach_point(center, ori, scene_size)
            shelves[s["id"]] = {
                "role": role, "center": center,
                "products": s.get("products", []),
                "n_products": s.get("product_count", 0),
                "approach": ap, "face_deg": face,
            }

    # 休息区起点: 默认放商业区左侧走廊(机器人"待命"位), 可由 --rest-area 覆盖
    if rest_area is None:
        rest_area = [2.0, round(scene_size[1] / 2, 2)]
    robot = {"pos": [round(rest_area[0], 2), round(rest_area[1], 2)], "yaw_deg": 0.0}

    waypoints = {"rest_area": robot["pos"]}
    for sid, s in shelves.items():
        waypoints[f"{sid}_approach"] = s["approach"]
    return {"scene_size": scene_size, "robot": robot,
            "waypoints": waypoints, "shelves": shelves}


def _build_map_block(facts):
    """把 facts 拼成喂给 VLM 的文本地图(对齐 deploy 的 full_prompt_block 风格)。

    若货架带 role(warehouse/commercial), 按 restock 语义分区展示:
    仓库货架=取货源, 商业货架=补货目标。否则退回通用展示(sim 单货架路径)。
    """
    has_roles = any("role" in s for s in facts["shelves"].values())
    L = ["=== STORE MAP (top-down image 'topdown.png' shows all shelves with IDs + coords) ===",
         f"Scene size: {facts['scene_size'][0]:.1f}m x {facts['scene_size'][1]:.1f}m", ""]

    if has_roles:
        L.append("=== WAREHOUSE SHELVES (stock source — pick items FROM here) ===")
        for sid, s in facts["shelves"].items():
            if s.get("role") != "warehouse" or s.get("n_products", 0) <= 0:
                continue
            prods = sorted({p.split(".")[-1] for p in s["products"]})
            L.append(f"  {sid} at ({s['center'][0]:.2f},{s['center'][1]:.2f}) "
                     f"approach=({s['approach'][0]:.2f},{s['approach'][1]:.2f}) "
                     f"face={s['face_deg']:.0f}deg: {', '.join(prods[:6])}"
                     + (f" (+{len(prods)-6})" if len(prods) > 6 else ""))
        L += ["", "=== COMMERCIAL SHELVES (restock target — place items ONTO here) ==="]
        for sid, s in facts["shelves"].items():
            if s.get("role") != "commercial":
                continue
            prods = sorted({p.split(".")[-1] for p in s.get("products", [])})
            desc = f": {', '.join(prods[:6])}" if prods else " (empty / needs stock)"
            L.append(f"  {sid} at ({s['center'][0]:.2f},{s['center'][1]:.2f}) "
                     f"approach=({s['approach'][0]:.2f},{s['approach'][1]:.2f}) "
                     f"face={s['face_deg']:.0f}deg{desc}")
    else:
        L.append("=== SHELF INVENTORY (only stocked shelves listed) ===")
        stocked = False
        for sid, s in facts["shelves"].items():
            if s.get("n_products", 0) <= 0:
                continue
            stocked = True
            L.append(f"\n{sid} at ({s['center'][0]:.2f}, {s['center'][1]:.2f}):")
            if s.get("layers"):
                for b, items in s["layers"].items():
                    L.append(f"  layer_{b}: {', '.join(items)}")
            else:
                L.append(f"  {s['n_products']} items")
        if not stocked:
            L.append("  (no stocked shelves)")

    L += ["", "=== NAVIGABLE WAYPOINTS (use these EXACT coords as navigate_to targets) ==="]
    for wp, xy in facts["waypoints"].items():
        L.append(f"  {wp}: ({xy[0]:.2f}, {xy[1]:.2f})")
    r = facts["robot"]
    L += ["", f"=== ROBOT NOW: at ({r['pos'][0]:.2f}, {r['pos'][1]:.2f}) "
          f"facing {r['yaw_deg']:.0f}deg ==="]
    return "\n".join(L)


# ============================================================
# 2. 落盘 VLM 输入(俯视图 + 文本地图),供人工核对
# ============================================================
def dump_vlm_inputs(map_block, facts, png_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 俯视图已由 generate_topdown_map 写入 png_path, 这里只确认
    if png_path and os.path.exists(png_path):
        print(f"[dump] topdown image -> {png_path}")
    else:
        print("[dump] WARN topdown image unavailable")

    # 文本地图(喂给 VLM 的 prompt block)
    txt_path = os.path.join(out_dir, "map_block.txt")
    with open(txt_path, "w") as f:
        f.write(map_block)
    print(f"[dump] map_block text -> {txt_path}")

    # 结构化坐标/物品(供规划校验)
    facts_path = os.path.join(out_dir, "scene_facts.json")
    with open(facts_path, "w") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)
    print(f"[dump] scene facts -> {facts_path}")


# ============================================================
# 3. 规划校验 — 不执行, 只检查规划的"正确性"
# ============================================================
def validate_plan(plan, facts, command, target_product="", nav_nodes=None):
    """检查 VLM 规划是否合理。返回 (passed: bool, checks: list[dict])。

    校验维度:
      1. 计划非空
      2. 子任务类型全部合法
      3. navigate_to 坐标取自已知 waypoint / 货架接近点 / 路口(不是 VLM 瞎编)
      4. 目标物品(如指定)确实在某个货架清单里
      5. 序列合理: 至少一次 navigate_to 出现在第一个操作子任务之前
    """
    VALID = {"pick_to_basket", "restock_basket_to_shelf", "pick_from_floor",
             "navigate_to", "turn_to"}
    # 已知坐标集合(waypoint + 货架接近点 + 货架中心 + 路口), 四舍五入到 0.1m 容差
    known = set()
    for xy in facts["waypoints"].values():
        known.add((round(xy[0], 1), round(xy[1], 1)))
    for s in facts["shelves"].values():
        known.add((round(s["center"][0], 1), round(s["center"][1], 1)))
        if "approach" in s:
            known.add((round(s["approach"][0], 1), round(s["approach"][1], 1)))
    # 像素路径方案下, nav 坐标是吸附后的合法路口 -> 纳入已知点
    if nav_nodes:
        for xy in nav_nodes.values():
            known.add((round(xy[0], 1), round(xy[1], 1)))
    all_products = [p.lower() for s in facts["shelves"].values()
                    for p in s.get("products", [])]

    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    add("plan_non_empty", len(plan) > 0, f"{len(plan)} subtasks")

    bad_types = [s["type"] for s in plan if s["type"] not in VALID]
    add("types_valid", not bad_types, f"invalid: {bad_types}" if bad_types else "all valid")

    # navigate_to 坐标必须取自已知点(容差 0.1m)
    bad_coords = []
    for s in plan:
        if s["type"] != "navigate_to":
            continue
        try:
            x, y = [float(v) for v in str(s["target"]).split(",")]
        except (ValueError, AttributeError):
            bad_coords.append(s["target"]); continue
        if (round(x, 1), round(y, 1)) not in known:
            bad_coords.append(s["target"])
    add("nav_coords_from_known_points", not bad_coords,
        f"invented/unmatched: {bad_coords}" if bad_coords else "all from waypoints/approach pts")

    # 目标物品在清单里
    if target_product:
        in_inv = target_product.lower() in " ".join(all_products)
        add("target_product_in_inventory", in_inv,
            f"'{target_product}' {'found' if in_inv else 'NOT found'} in shelves")

    # 序列合理: 第一个操作动作前必须先有导航
    op_types = {"pick_to_basket", "restock_basket_to_shelf", "pick_from_floor"}
    first_op = next((i for i, s in enumerate(plan) if s["type"] in op_types), None)
    if first_op is not None:
        has_nav_before = any(s["type"] == "navigate_to" for s in plan[:first_op])
        add("navigate_before_manipulation", has_nav_before,
            "nav precedes first manipulation" if has_nav_before
            else "manipulation without navigating first")

    passed = all(c["pass"] for c in checks)
    return passed, checks


# ============================================================
# 4. (可选) 调 VLM 规划器 — 直接复用同目录的 deploy_supermarket.VLMPlanner
# ============================================================
def expand_nav_with_astar(plan, facts, obstacles):
    """用 A* 重建导航: VLM 选货架序列, 代码算每段沿通道不撞货架的全局路点。

    丢弃 VLM 自己报的 navigate_to(可能穿货架), 改为: 依操作动作针对的货架取其
    approach 点, 从机器人当前位置用 A* 在中轴路网上求路径, 展开成 navigate_to 链。
    返回 (新plan, 路网nodes, 邻接adj, A*整条节点路径 path_nodes, approach点表)。
    """
    from corridor_graph import (medial_axis_nodes, merge_close_joints,
                                prune_unsafe_edges, connect_components,
                                attach_point, astar)
    scene = list(facts["scene_size"])
    robot_xy = [round(float(facts["robot"]["pos"][0]), 2),
                round(float(facts["robot"]["pos"][1]), 2)]
    # 中轴法: 只保留分叉/端点为路口, 边沿骨架追踪相邻路口直连(已验证不穿膨胀货架)。
    # 此店是商业+仓库两块错开子网格, 中线法投影无干净空缝 -> 必须用中轴法。
    # 膨胀 0.6m: 比机器人半径更宽, 让中线更靠通道正中, 远离货架。
    nodes, edges, skel = medial_axis_nodes(obstacles, scene, sample_step=0.8,
                                           robot_radius=0.6,
                                           key_points={"REST": robot_xy},
                                           return_skeleton=True)
    adj = {n: [] for n in nodes}
    name_of = {tuple(xy): nm for nm, xy in nodes.items()}
    import numpy as _np
    for (p0, p1) in edges:
        a, b = name_of.get(tuple(p0)), name_of.get(tuple(p1))
        if a and b and a != b:
            d = float(_np.hypot(p0[0] - p1[0], p0[1] - p1[1]))
            adj[a].append((b, d))
            adj[b].append((a, d))
    # 消簇: 合并 <0.5m 的近邻路口; 传 obstacles 使合并后的新边验证不穿膨胀货架
    nodes, adj = merge_close_joints(nodes, adj, radius=0.5, keep={"REST"},
                                    obstacles=obstacles, robot_radius=0.6)
    # 最终安全兜底: 删掉所有穿膨胀货架(0.6m)的边, 硬保证不贴/撞货架
    adj, _bad = prune_unsafe_edges(nodes, adj, obstacles, robot_radius=0.6)
    if _bad:
        print(f"  [nav] 兜底删除 {_bad} 条穿货架边")
    # 连通性修复: medial axis 在开阔区可能断开 -> 用安全直线边把孤立块接回主块
    adj, _added = connect_components(nodes, adj, obstacles, robot_radius=0.6)
    if _added:
        print(f"  [nav] 连通修复新增 {_added} 条安全边")

    def shelf_of(subtask):
        """操作动作 -> 它针对的货架 id(从 target/instruction 里找)。"""
        t = str(subtask.get("target", "")) + " " + str(subtask.get("instruction", ""))
        for sid in facts["shelves"]:
            if sid in t:
                return sid
        return None

    op = {"pick_to_basket", "restock_basket_to_shelf", "pick_from_floor"}

    # 只接入任务真正用到的货架 approach 点(不是全部 25 个), 每个只连最近 1 个无障碍路点
    needed = []
    for s in plan:
        if s.get("type") in op:
            sid = shelf_of(s)
            if sid and sid not in needed:
                needed.append(sid)
    app_pts = {}
    for sid in needed:
        ap = facts["shelves"].get(sid, {}).get("approach")
        if ap:
            nm = f"{sid}_app"
            attach_point(nodes, adj, obstacles, nm, ap,
                         robot_radius=0.6, max_connect=3.0, max_edges=1)
            app_pts[nm] = [round(ap[0], 2), round(ap[1], 2)]

    new_plan, path_nodes = [], []
    legs = []  # 每段 A* 路径(节点名列表): leg0=去取货, leg1=去补货 ...
    cur = "REST"
    for s in plan:
        if s.get("type") == "navigate_to":
            continue  # 丢弃 VLM 的导航, 由 A* 重建
        if s.get("type") in op:
            sid = shelf_of(s)
            goal = f"{sid}_app" if sid and f"{sid}_app" in nodes else None
            if goal and goal != cur:
                seg = astar(nodes, adj, cur, goal)
                if seg:
                    legs.append(seg)
                    path_nodes += seg if not path_nodes else seg[1:]
                    for nm in seg[1:]:
                        x, y = nodes[nm]
                        new_plan.append({"type": "navigate_to",
                                         "instruction": f"沿通道行驶至 {nm}",
                                         "target": f"{x},{y}"})
                    cur = goal
        new_plan.append(s)
    return new_plan, nodes, adj, path_nodes, app_pts, skel, legs


def expand_nav_with_grid(plan, facts, obstacles, res=0.1, robot_radius=0.1,
                         inflation_radius=1.5):
    """栅格 + 梯度膨胀 + A* 重建导航(拓扑图的替代 planner)。

    与 expand_nav_with_astar 返回值同构 -> 可视化/NavDP 完全不用改:
      (新plan, nodes, adj, path_nodes, app_pts, skel, legs)。
    机制不同: 这里不建路网图, 而是对每段(当前位置->货架approach)直接栅格 A*+抽稀
    出世界路点; 再把所有路点合成为等价链式 nodes/adj 供原可视化绘制。

    梯度膨胀让 A* 走通道正中; 贴货架的 approach 点经目标豁免保证可达(最后一小段贴障
    交给 NavDP 局部避障兜)。
    """
    from grid_planner import GridPlanner

    scene = list(facts["scene_size"])
    robot_xy = [round(float(facts["robot"]["pos"][0]), 2),
                round(float(facts["robot"]["pos"][1]), 2)]
    planner = GridPlanner(obstacles, scene, res=res, robot_radius=robot_radius,
                          inflation_radius=inflation_radius)

    def shelf_of(subtask):
        t = str(subtask.get("target", "")) + " " + str(subtask.get("instruction", ""))
        for sid in facts["shelves"]:
            if sid in t:
                return sid
        return None

    op = {"pick_to_basket", "restock_basket_to_shelf", "pick_from_floor"}

    # 需要到达的货架 approach 点(任务真正用到的)
    needed = []
    for s in plan:
        if s.get("type") in op:
            sid = shelf_of(s)
            if sid and sid not in needed:
                needed.append(sid)
    app_pts = {}
    for sid in needed:
        ap = facts["shelves"].get(sid, {}).get("approach")
        if ap:
            app_pts[f"{sid}_app"] = [round(ap[0], 2), round(ap[1], 2)]

    # 合成等价路网(节点=所有路点, 边=每段内相邻路点连线), 供原可视化
    nodes = {"REST": robot_xy}
    adj = {"REST": []}
    nid = [0]

    def _add_node(xy):
        nm = f"G{nid[0]}"; nid[0] += 1
        nodes[nm] = [round(float(xy[0]), 2), round(float(xy[1]), 2)]
        adj.setdefault(nm, [])
        return nm

    def _link(a, b):
        import numpy as _np
        d = float(_np.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1]))
        adj[a].append((b, d)); adj[b].append((a, d))

    new_plan, path_nodes, legs = [], [], []
    cur_name, cur_xy = "REST", robot_xy
    for s in plan:
        if s.get("type") == "navigate_to":
            continue  # 丢弃 VLM 的导航, 由栅格 A* 重建
        if s.get("type") in op:
            sid = shelf_of(s)
            goal_app = f"{sid}_app" if sid and f"{sid}_app" in app_pts else None
            if goal_app:
                goal_xy = app_pts[goal_app]
                waypts = planner.plan(cur_xy, goal_xy)  # 世界路点序列(含首尾)
                if waypts and len(waypts) >= 2:
                    # 合成节点链: cur_name -> 中间路点 -> goal_app
                    seg = [cur_name]
                    for wp in waypts[1:-1]:
                        nm = _add_node(wp)
                        _link(seg[-1], nm)
                        seg.append(nm)
                    # 终点用 approach 点名(与可视化 app_pts 一致)
                    nodes[goal_app] = goal_xy
                    adj.setdefault(goal_app, [])
                    _link(seg[-1], goal_app)
                    seg.append(goal_app)
                    legs.append(seg)
                    path_nodes += seg if not path_nodes else seg[1:]
                    for nm in seg[1:]:
                        x, y = nodes[nm]
                        new_plan.append({"type": "navigate_to",
                                         "instruction": f"沿通道行驶至 {nm}",
                                         "target": f"{x},{y}"})
                    cur_name, cur_xy = goal_app, goal_xy
        new_plan.append(s)

    skel = None  # 栅格无中轴骨架; 可视化对 skel=None 安全
    return new_plan, nodes, adj, path_nodes, app_pts, skel, legs


def resolve_pixel_nav(plan, nodes, img_wh, scene_size):
    """把 VLM 的像素 navigate_to 目标转世界坐标 + 匹配最近路口, 改写 target 为 "x,y"。

    VLM 现在按提示词输出 navigate_to target="px,py"(图像像素)。这里:
      像素 -> pixel_to_world -> nearest_node 吸附到合法路口 -> 写回世界坐标。
    非 navigate_to 子任务原样保留。返回 (新plan, 调试信息列表)。
    """
    from corridor_graph import pixel_to_world, nearest_node
    new_plan, dbg = [], []
    for s in plan:
        if s.get("type") != "navigate_to":
            new_plan.append(s)
            continue
        try:
            px, py = [float(v) for v in str(s["target"]).split(",")]
        except (ValueError, KeyError):
            new_plan.append(s)
            continue
        world = pixel_to_world(px, py, img_wh, scene_size)
        name, node_xy = nearest_node(nodes, world)
        ns = dict(s)
        ns["target"] = f"{node_xy[0]},{node_xy[1]}"
        new_plan.append(ns)
        dbg.append({"pixel": [px, py], "world": world, "node": name, "node_xy": node_xy})
    return new_plan, dbg


def call_vlm_plan(command, map_block, topdown_image, args):
    """复用本项目 deploy_supermarket.py 里的 VLMPlanner, 返回 [{type,instruction,target}, ...]。"""
    from deploy_supermarket import VLMPlanner

    # 默认用环境里已有的 Anthropic 凭据(ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN)
    api_key = args.vlm_api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    base_url = args.vlm_base_url or os.environ.get("ANTHROPIC_BASE_URL", "")
    # 模型: 命令行 > 环境 ANTHROPIC_MODEL(packyAPI aws-q 分组只有 claude-opus-4-8)
    # > VLMPlanner 内置默认
    model = args.vlm_model
    if not model and args.vlm_provider == "anthropic":
        # packyAPI aws-q 分组只有 claude-opus-4-8; 不能落到 VLMPlanner 内置的
        # sonnet 默认(该模型在此代理不存在 -> 返回空/非文本响应)。
        model = os.environ.get("ANTHROPIC_MODEL", "") or "claude-opus-4-8"
    planner = VLMPlanner(provider=args.vlm_provider, api_key=api_key,
                         model=model, base_url=base_url)
    print(f"[vlm] provider={args.vlm_provider} model={planner.model} "
          f"base_url={base_url or '(default)'}")
    subtasks = planner.plan(command, map_block, topdown_image)
    return [s.to_dict() for s in subtasks]


# ============================================================
# 5. Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="VLM planning-only test on a RoboBenchMart scene")
    p.add_argument("--scene-dir", required=True, help="场景目录(含 input_config.yaml)")
    p.add_argument("--env-name", default="PickToBasketContNiveaEnv", help="注册的环境名")
    p.add_argument("--robot-uids", default="ds_fetch_basket")
    p.add_argument("--sim-backend", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--cam-height", type=float, default=8.0, help="俯视相机高度(米)")
    p.add_argument("--image-size", type=int, default=1024, help="俯视图分辨率")
    p.add_argument("--out-dir", default="vlm_plan_test", help="输出目录")
    # 多货架 restock 规划: 不起 sim, 直接读 layout(绕过 Cont 环境单货架限制)
    p.add_argument("--from-layout", action="store_true",
                   help="不起 sim, 直接从场景 layout 构建地图(支持多货架 restock 场景)")
    p.add_argument("--rest-area", default="",
                   help="休息区起点 'x,y'(仅 --from-layout); 默认商业区左侧走廊")
    # VLM 规划(可选)
    p.add_argument("--command", default="", help="高层指令; 配合 --call-vlm")
    p.add_argument("--target-product", default="", help="目标物品名, 用于校验是否在清单里")
    p.add_argument("--call-vlm", action="store_true", help="真正调 VLM 规划并校验")
    p.add_argument("--vlm-provider", default="anthropic",
                   choices=["qwen", "gemini", "openai", "openai_compatible", "anthropic"])
    p.add_argument("--vlm-api-key", default="")
    p.add_argument("--vlm-model", default="")
    p.add_argument("--vlm-base-url", default="")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir = os.path.abspath(args.out_dir)

    if args.from_layout:
        # 不起 sim: 直接读 layout 画全场景图 + 构建多货架 restock facts。
        # 绕过 Cont 操作环境"只支持 1 个 active 货架"的设计约束(规划测试无需 sim)。
        os.makedirs(args.out_dir, exist_ok=True)
        sys.path.insert(0, os.path.join(RBM_ROOT, "scripts"))
        from generate_topdown_map import generate_map
        scene_dir = args.scene_dir if os.path.isabs(args.scene_dir) \
            else os.path.join(RBM_ROOT, args.scene_dir)
        png_path = os.path.join(args.out_dir, "topdown.png")
        rest = None
        if args.rest_area:
            rest = [float(v) for v in args.rest_area.split(",")]
        _, coords_json = generate_map(
            scene_dir, output_path=png_path,
            title=f"{os.path.basename(scene_dir)} — Restock Planning Map",
            show_zones=True, show_coord_table=False)
        facts = _build_facts_from_layout(coords_json, rest_area=rest)
        if rest is None:
            facts["robot"]["pos"] = facts["waypoints"]["rest_area"]
        env = None
        map_block = _build_map_block(facts)
        print(f"[map] {len(facts['shelves'])} shelves (no sim), "
              f"rest_area at {facts['robot']['pos']}")
    else:
        # env 必须保留引用(避免被 GC 回收, 否则场景销毁)
        env, map_block, facts, png_path = build_scene_and_extract(
            args.scene_dir, args.env_name, args.robot_uids, args.sim_backend,
            args.out_dir)

    dump_vlm_inputs(map_block, facts, png_path, args.out_dir)

    print("\n" + "=" * 60)
    print("VLM 文本输入预览 (map_block.txt):")
    print("=" * 60)
    print(map_block)

    if not args.call_vlm:
        print("\n[done] 仅抽取 VLM 输入 (未调用 VLM)。加 --command ... --call-vlm 跑规划+校验。")
        return

    if not args.command:
        print("\n[error] --call-vlm 需要同时给 --command")
        return

    print("\n" + "=" * 60)
    print(f"调 VLM 规划: {args.command}")
    print("=" * 60)

    # A* 导航: VLM 看带货架标签的 topdown.png 选货架, 代码用中轴路网+A* 算路径。
    obstacles = None
    if env is not None:
        from corridor_graph import obstacles_from_layout
        scene_dir_abs = args.scene_dir if os.path.isabs(args.scene_dir) \
            else os.path.join(RBM_ROOT, args.scene_dir)
        obstacles = obstacles_from_layout(scene_dir_abs, env=env)
        print(f"[nav] {len(obstacles)} obstacles from layout (含仓库整排); A* will plan routes")

    try:
        topdown_img = None
        if png_path and os.path.exists(png_path):
            import cv2
            bgr = cv2.imread(png_path)
            if bgr is not None:
                topdown_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        plan = call_vlm_plan(args.command, map_block, topdown_img, args)
    except Exception as e:
        print(f"[error] VLM 调用失败: {e}")
        return

    # A* 重建导航: 丢弃 VLM 自报的 navigate_to, 按其选的货架序列算沿通道不撞货架的路点
    nav_nodes = None
    if obstacles is not None:
        plan, nav_nodes, nav_adj, path_nodes, app_pts, skel, legs = \
            expand_nav_with_astar(plan, facts, obstacles)
        print(f"  [nav] A* 路径 {len(path_nodes)} 个路口, 接入 {len(app_pts)} 个货架操作位")

    print("\n规划结果:")
    for i, s in enumerate(plan):
        print(f"  [{i+1}] {s['type']:24s} target={s.get('target',''):16s} | {s['instruction']}")

    passed, checks = validate_plan(plan, facts, args.command, args.target_product,
                                   nav_nodes=nav_nodes)

    print("\n校验:")
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']:32s} {c['detail']}")
    print("\n" + ("PLAN VALID ✅" if passed else "PLAN HAS ISSUES ❌"))

    out = {"command": args.command, "passed": passed, "plan": plan, "checks": checks}
    plan_path = os.path.join(args.out_dir, "plan.json")
    with open(plan_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] plan + checks -> {plan_path}")

    # 可视化: 在原始 topdown.png(带货架标签)同坐标系内画 A* 路径(world坐标->严格对齐)
    # 图1 nav_route: 第1段(去取货)红色 + 第2段(去补货)蓝色
    # 图2 nav_graph: 实际用到的节点+边叠在原图上
    if obstacles is not None and nav_nodes is not None and path_nodes:
        from generate_topdown_map import generate_map as _gm
        scene_dir = args.scene_dir if os.path.isabs(args.scene_dir) \
            else os.path.join(RBM_ROOT, args.scene_dir)
        robot = facts["robot"]
        active_pos = None
        for s in facts["shelves"].values():
            if s.get("role") == "commercial" and "approach" in s and s["center"][0] > 0.01:
                active_pos = s["center"]
                break

        def leg_xy(i):
            return [nav_nodes[n] for n in legs[i] if n in nav_nodes] if i < len(legs) else None

        leg1_xy = leg_xy(0)              # REST -> 取货货架(红)
        leg2_xy = leg_xy(1)              # 取货货架 -> 补货货架(蓝)
        route = os.path.join(args.out_dir, "nav_route.png")
        _gm(scene_dir, output_path=route,
            robot_angle=robot["yaw_deg"], robot_pos=robot["pos"],
            active_pos=active_pos, title=f"{args.command[:24]} — 红:取货 蓝:补货",
            show_zones=False, show_coord_table=False,
            nav_path=leg1_xy, nav_path2=leg2_xy)
        print(f"[viz] 两段路线(红取货/蓝补货)-> {route}")

        # 图2: 实际用到的节点 + 边(沿 A* 路径相邻节点)叠原图
        used = list(dict.fromkeys(path_nodes))   # 去重保序
        node_list = [{"id": n, "x": nav_nodes[n][0], "y": nav_nodes[n][1],
                      "type": ("goal" if n in app_pts else
                               "rest_area" if n == "REST" else "cross")}
                     for n in used if n in nav_nodes]
        edges = [(nav_nodes[path_nodes[i]], nav_nodes[path_nodes[i + 1]])
                 for i in range(len(path_nodes) - 1)
                 if path_nodes[i] in nav_nodes and path_nodes[i + 1] in nav_nodes]
        graph = os.path.join(args.out_dir, "nav_graph.png")
        _gm(scene_dir, output_path=graph,
            robot_angle=robot["yaw_deg"], robot_pos=robot["pos"],
            active_pos=active_pos, title=f"{args.command[:24]} — 路网节点+边",
            show_zones=False, show_coord_table=False,
            nav_nodes=node_list, nav_edges=edges)
        print(f"[viz] {len(node_list)} 节点 + {len(edges)} 边叠原图 -> {graph}")

        # 图3: 完整路网(所有节点 + 所有边)叠原图 — 不只 A* 用到的
        all_nodes = [{"id": n, "x": nav_nodes[n][0], "y": nav_nodes[n][1],
                      "type": ("goal" if n in app_pts else
                               "rest_area" if n == "REST" else "cross")}
                     for n in nav_nodes]
        seen_e = set()
        all_edges = []
        for a, nbrs in nav_adj.items():
            for b, _ in nbrs:
                key = (min(a, b), max(a, b))
                if key in seen_e or a not in nav_nodes or b not in nav_nodes:
                    continue
                seen_e.add(key)
                all_edges.append((nav_nodes[a], nav_nodes[b]))
        full = os.path.join(args.out_dir, "nav_graph_full.png")
        _gm(scene_dir, output_path=full,
            robot_angle=robot["yaw_deg"], robot_pos=robot["pos"],
            active_pos=active_pos, title=f"{args.command[:24]} — 完整路网",
            show_zones=False, show_coord_table=False,
            nav_nodes=all_nodes, nav_edges=all_edges)
        print(f"[viz] 完整路网 {len(all_nodes)} 节点 + {len(all_edges)} 边叠原图 -> {full}")


if __name__ == "__main__":
    main()




