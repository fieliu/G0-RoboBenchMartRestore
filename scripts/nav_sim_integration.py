"""把 A* 路网导航集成进仿真(两阶段) + 第三人称录视频。

架构(你要的): A*(全局通道路点) -> 逐个路点送入 NavDP(局部避障) -> NavDP 内部
控制器输出底盘速度 -> 驱动机器人 -> 到货架 approach 位后转向货架朝向。
  阶段1: REST -> 仓库取货货架(leg0 路点序列)-> 转向货架
  阶段2: 取货货架 -> 商业区补货货架(leg1 路点序列)-> 转向货架
跳过 VLA 操作与完成判定, 只验证导航是否走通。

复用: NavDPController(已验证的 NavDP 接口) + run_navdp_nav.py 的底盘动作构造/
第三人称相机/录像(pd_joint_pos, 13维 action, base 在末两维) + A* 建图(test_vlm_planning)。
"""
import argparse, os, sys, time
import numpy as np, cv2
import gymnasium as gym, sapien

RBM_ROOT = os.environ.get("RBM_ROOT", "/home/lh/VLA/RoboBenchMart-main")
NAVDP_CKPT = os.path.join(RBM_ROOT, "dsynth/navigation/navdp_models/navdp-cross-modal.ckpt")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, RBM_ROOT)
sys.path.insert(0, os.path.join(RBM_ROOT, "scripts"))

import dsynth.envs   # noqa: F401  注册环境
import dsynth.robots  # noqa: F401
from dsynth.navigation.navdp_controller import NavDPController
from mani_skill.utils import sapien_utils

# ── 底层 helpers (复用 run_navdp_nav.py 的实现, pd_joint_pos / 13维 action) ──────
def _arm_qpos(env):  return env.unwrapped.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
def _body_qpos(env): return env.unwrapped.agent.controller.controllers["body"].qpos[0].cpu().numpy()
def get_depth(obs, cam="head_camera"): return obs["sensor_data"][cam]["depth"][0].cpu().numpy()
def get_rgb(obs, cam="head_camera"):   return obs["sensor_data"][cam]["rgb"][0].cpu().numpy()
def get_robot_pose_matrix(env):
    return env.unwrapped.agent.base_link.pose.to_transformation_matrix()[0].cpu().numpy()

def make_action(env, lv, av):
    """7 arm + 1 gripper + 3 body + [lv, av] = 13 维; pd_joint_pos 下保持当前臂/身姿态。"""
    return np.hstack([_arm_qpos(env), 0.015, _body_qpos(env), [lv, av]]).astype(np.float32)

def update_camera(env, rp, distance=4.0, height=2.4, look_ahead=2.5):
    """第三人称跟随相机: 机器人后上方 distance/height, 看向前方 look_ahead。"""
    p, f = rp[:3, 3], rp[:3, 0]
    src = p - distance * f + np.array([0, 0, height])
    tgt = p + look_ahead * f + np.array([0, 0, 0.8])
    n = sapien_utils.look_at(src, tgt).raw_pose[0].cpu().numpy()
    sc = env.unwrapped.scene.human_render_cameras["render_camera"].camera._render_cameras[0]
    sc.set_local_pose(sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
    sc.set_entity_pose(sapien.Pose(p=n[:3], q=n[3:]))

def write_frame(vw, env, obs, text_lines=None):
    """第三人称 RGB 帧写入视频(可叠加文字)。"""
    frame = env.render()
    if frame is None or vw is None:
        return
    fb = frame if isinstance(frame, np.ndarray) else frame[0].cpu().numpy()
    fb = cv2.cvtColor(fb, cv2.COLOR_RGB2BGR)
    if text_lines:
        for i, txt in enumerate(text_lines):
            cv2.putText(fb, txt, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
    vw.write(fb)

def robot_xy_yaw(env):
    T = get_robot_pose_matrix(env)
    fwd = T[:3, 0]
    return float(T[0, 3]), float(T[1, 3]), float(np.arctan2(fwd[1], fwd[0]))

def drive_to_waypoint_navdp(env, navdp, goal_xy, vw, *, pos_tol, replan_every=4,
                            max_steps=400, frame_every=3, label=""):
    """用 NavDP 把底盘开到一个世界路点 (x,y)。NavDP 内部局部避障 + pure-pursuit 出速度。

    每 replan_every 步用当前 RGB-D 重规划一次(NavDP 有状态), 到达(<pos_tol)返回 True。
    """
def drive_to_waypoint_navdp(env, navdp, goal_xy, vw, obs, *, pos_tol, replan_every=4,
                            max_steps=400, frame_every=3, label=""):
    """用 NavDP 把底盘开到一个世界路点 (x,y)。NavDP 内部局部避障 + pure-pursuit 出速度。

    obs 为上一次 env.step/reset 的观测(含 sensor_data); 每步用它的 RGB-D 规划。
    每 replan_every 步重规划一次(NavDP 有状态)。返回 (到达?, 最新 obs)。
    """
    navdp.reset()
    for i in range(max_steps):
        rp = get_robot_pose_matrix(env)
        x, y = rp[0, 3], rp[1, 3]
        dist = float(np.hypot(goal_xy[0] - x, goal_xy[1] - y))
        if dist < pos_tol:
            return True, obs
        rgb = get_rgb(obs)
        depth = get_depth(obs)
        goal_world = np.array([goal_xy[0], goal_xy[1], 0.0])
        goal_robot = navdp.compute_goal_in_robot_frame(goal_world, rp)
        if i % replan_every == 0 or navdp.traj_world is None:
            navdp.plan(rgb, depth, goal_robot, rp)
        lv, av = navdp.compute_base_velocity(rp)
        obs, _, _, _, _ = env.step(make_action(env, lv, av))
        update_camera(env, get_robot_pose_matrix(env))
        if vw is not None and i % frame_every == 0:
            write_frame(vw, env, obs, [f"{label} NavDP->({goal_xy[0]:.1f},{goal_xy[1]:.1f}) d={dist:.2f}"])
    return False, obs

def turn_to_face(env, face_deg, vw, *, tol_deg=5.0, kp=1.2, max_steps=200,
                 frame_every=3, label=""):
    """到位后原地转到货架朝向 face_deg(度)。yaw P 控制。"""
    target = np.radians(face_deg)
    for i in range(max_steps):
        _, _, yaw = robot_xy_yaw(env)
        err = float((target - yaw + np.pi) % (2 * np.pi) - np.pi)
        if abs(err) < np.radians(tol_deg):
            return True
        av = float(np.clip(kp * err, -0.7, 0.7))
        o, _, _, _, _ = env.step(make_action(env, 0.0, av))
        update_camera(env, get_robot_pose_matrix(env))
        if vw is not None and i % frame_every == 0:
            write_frame(vw, env, o, [f"{label} TURN->{face_deg:.0f}deg err={np.degrees(err):.0f}"])
    return False

def build_args_for_vlm(command, scene_dir, env_name, out_dir):
    import argparse as _ap
    a = _ap.Namespace()
    a.command = command; a.scene_dir = scene_dir; a.env_name = env_name
    a.out_dir = out_dir; a.vlm_provider = "anthropic"
    a.vlm_api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    a.vlm_base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    a.vlm_model = os.environ.get("ANTHROPIC_MODEL", "")
    a.target_product = ""
    return a

def shelf_face_deg(facts, sid):
    """货架 approach 朝向(度); 没有则 None。"""
    s = facts.get("shelves", {}).get(sid, {})
    return s.get("face_deg")

def run(args):
    import test_vlm_planning as T
    from corridor_graph import obstacles_from_layout

    scene_dir = args.scene_dir if os.path.isabs(args.scene_dir) \
        else os.path.join(RBM_ROOT, args.scene_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 场景 + facts + topdown
    env, map_block, facts, png_path = T.build_scene_and_extract(
        scene_dir, args.env_name, args.robot_uids, args.sim_backend, args.out_dir)
    obstacles = obstacles_from_layout(scene_dir, env=env)

    # 2) VLM 规划一次
    topdown_img = None
    if png_path and os.path.exists(png_path):
        bgr = cv2.imread(png_path)
        if bgr is not None:
            topdown_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    va = build_args_for_vlm(args.command, args.scene_dir, args.env_name, args.out_dir)
    print(f"[vlm] 规划: {args.command}")
    plan = T.call_vlm_plan(args.command, map_block, topdown_img, va)

    # 3) A* 出路点 + 分段(legs: leg0=去取货, leg1=去补货)
    if getattr(args, "planner", "topo") == "grid":
        print("[nav] planner=grid (栅格+梯度膨胀+A*)")
        plan, nodes, adj, path_nodes, app_pts, skel, legs = \
            T.expand_nav_with_grid(plan, facts, obstacles,
                                   res=args.grid_res,
                                   robot_radius=args.robot_radius,
                                   inflation_radius=args.inflation_radius)
    else:
        print("[nav] planner=topo (拓扑图+A*)")
        plan, nodes, adj, path_nodes, app_pts, skel, legs = \
            T.expand_nav_with_astar(plan, facts, obstacles)
    if not legs:
        print("[error] A* 没生成任何段, 退出"); env.close(); return
    print(f"[nav] {len(legs)} 段; 路点序列:")
    for i, lg in enumerate(legs):
        print(f"   leg{i}: {lg}")

    # 存这次的 plan + 操作子任务数(便于看为什么少一段) + 画导航图
    _save_plan_and_viz(plan, facts, nodes, adj, path_nodes, app_pts, legs, scene_dir, args)

    # 每段终点的货架 id(用于到位后转向 face_deg)
    def shelf_of_app(app_name):
        return app_name[:-4] if app_name.endswith("_app") else None

    # 4) NavDP + 录像
    navdp = NavDPController(model_path=args.navdp_ckpt, device=args.device)
    vid = os.path.join(args.out_dir, f"nav_sim_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    obs, _ = env.reset(options={"reconfigure": True})
    sf = env.render(); sf = sf if isinstance(sf, np.ndarray) else sf[0].cpu().numpy()
    vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30, (sf.shape[1], sf.shape[0]))
    obs = _relax_and_tilt(env, vw, obs)

    # 5) 两阶段: 每段逐路点喂 NavDP, 段尾转向货架
    ok_all = True
    for li, leg in enumerate(legs):
        stage = f"阶段{li+1}"
        mids, last = leg[1:-1], leg[-1]   # 跳过起点(=上段终点/REST)
        for wp in mids:
            xy = nodes[wp]
            ok, obs = drive_to_waypoint_navdp(env, navdp, xy, vw, obs, pos_tol=0.5,
                                              max_steps=args.max_steps_per_wp,
                                              label=f"{stage} 中途{wp}")
            print(f"  {stage} 路点 {wp}{xy}: {'到达' if ok else '超时'}")
        # 段终点(货架 approach): 严格阈值
        gxy = nodes[last]
        ok, obs = drive_to_waypoint_navdp(env, navdp, gxy, vw, obs, pos_tol=0.3,
                                          max_steps=args.max_steps_per_wp,
                                          label=f"{stage} 终点{last}")
        print(f"  {stage} 终点 {last}{gxy}: {'到达' if ok else '超时'}")
        if not ok:
            ok_all = False
        # 到位转向货架朝向
        sid = shelf_of_app(last)
        fd = shelf_face_deg(facts, sid) if sid else None
        if fd is not None:
            tok = turn_to_face(env, fd, vw, label=f"{stage} 面向{sid}")
            print(f"  {stage} 转向货架 {sid} face={fd:.0f}deg: {'完成' if tok else '超时'}")

    for _ in range(30):
        obs, _, _, _, _ = env.step(make_action(env, 0.0, 0.0))
        update_camera(env, get_robot_pose_matrix(env))
        write_frame(vw, env, obs, ["DONE" if ok_all else "TIMEOUT(部分)"])
    vw.release()
    print(f"[done] {'全部到达' if ok_all else '有路点超时'}  视频: {vid}")
    env.close()

def _relax_and_tilt(env, vw, obs):
    """导航姿态: 手臂垂放(不挡相机)。返回最新 obs。"""
    relax = np.array([0.0, 1.518, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    body = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(120):
        obs, _, _, _, _ = env.step(np.hstack([relax, 0.015, body, [0.0, 0.0]]).astype(np.float32))
        update_camera(env, get_robot_pose_matrix(env))
        if vw is not None and i % 4 == 0:
            write_frame(vw, env, obs, ["READY: arm hang-down"])
    return obs

def _save_plan_and_viz(plan, facts, nodes, adj, path_nodes, app_pts, legs, scene_dir, args):
    """存这次的 plan(便于看为什么少一段)+ 画导航图(路线/完整路网)。"""
    import json
    from generate_topdown_map import generate_map as _gm
    op = {"pick_to_basket", "restock_basket_to_shelf", "pick_from_floor"}
    ops = [s for s in plan if (s.get("type") if isinstance(s, dict) else getattr(s, "type", None)) in op]
    print(f"[plan] 操作子任务数={len(ops)} (补货任务应=2: 取货+补货)")
    for s in ops:
        d = s if isinstance(s, dict) else s.__dict__
        print(f"   {d.get('type')}: {str(d.get('instruction',''))[:50]!r}")
    try:
        with open(os.path.join(args.out_dir, "plan_navsim.json"), "w") as f:
            json.dump([s if isinstance(s, dict) else s.__dict__ for s in plan],
                      f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[plan] 存 plan 失败: {e}")

    robot = facts["robot"]
    # 图1: 两段路线(红取货/蓝补货)
    def leg_xy(i):
        return [nodes[n] for n in legs[i] if n in nodes] if i < len(legs) else None
    route = os.path.join(args.out_dir, "navsim_route.png")
    _gm(scene_dir, output_path=route, robot_angle=robot["yaw_deg"], robot_pos=robot["pos"],
        title=f"{args.command[:20]} — 红:取货 蓝:补货", show_zones=False, show_coord_table=False,
        nav_path=leg_xy(0), nav_path2=leg_xy(1))
    print(f"[viz] 路线图 -> {route}")
    # 图2: 完整路网(所有节点+边)
    all_nodes = [{"id": n, "x": nodes[n][0], "y": nodes[n][1],
                  "type": ("goal" if n in app_pts else "rest_area" if n == "REST" else "cross")}
                 for n in nodes]
    seen, all_edges = set(), []
    for a, nbrs in adj.items():
        for b, _ in nbrs:
            k = (min(a, b), max(a, b))
            if k in seen or a not in nodes or b not in nodes:
                continue
            seen.add(k); all_edges.append((nodes[a], nodes[b]))
    full = os.path.join(args.out_dir, "navsim_graph_full.png")
    _gm(scene_dir, output_path=full, robot_angle=robot["yaw_deg"], robot_pos=robot["pos"],
        title=f"{args.command[:20]} — 完整路网", show_zones=False, show_coord_table=False,
        nav_nodes=all_nodes, nav_edges=all_edges)
    print(f"[viz] 完整路网 {len(all_nodes)}节点 {len(all_edges)}边 -> {full}")

def main():
    p = argparse.ArgumentParser(description="A* 路点逐个喂 NavDP 的两阶段仿真导航 + 第三人称录像")
    p.add_argument("--scene-dir", default="generated_envs/restock_scene")
    p.add_argument("--env-name", default="RestockFlowContNiveaEnv")
    p.add_argument("--robot-uids", default="ds_fetch_basket")
    p.add_argument("--sim-backend", default="cpu",
                   help="cpu: PhysX GPU 在本机起不来; CPU 仿真 + CPU 渲染仍产出 sensor_data, NavDP 模型本身在 GPU 上跑")
    p.add_argument("--out-dir", default="restock_sim_plan")
    p.add_argument("--command", required=True, help="给 VLM 的自然语言指令")
    p.add_argument("--navdp-ckpt", default=NAVDP_CKPT, help="NavDP 权重 .ckpt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-steps-per-wp", type=int, default=400,
                   help="单路点 NavDP 驱动最大步数")
    p.add_argument("--planner", default="topo", choices=["topo", "grid"],
                   help="topo: 拓扑图+A*(默认); grid: 栅格+梯度膨胀+A*")
    p.add_argument("--grid-res", type=float, default=0.1, help="栅格分辨率(米/格)")
    p.add_argument("--robot-radius", type=float, default=0.1, help="机器人半径(米, 内切禁区)")
    p.add_argument("--inflation-radius", type=float, default=1.5,
                   help="梯度膨胀半径(米), 此距离外代价为0")
    args = p.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    run(args)

if __name__ == "__main__":
    main()
