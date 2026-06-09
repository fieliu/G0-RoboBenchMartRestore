"""走廊路口图 + A* 全局导航: 从 sim 实际障碍建路口图(点+验证过的边), A* 搜不穿货架的全局路径。

流程: obstacles_from_sim -> corridor_intersections(点) -> build_edges(验证过的边)
       -> astar(全局最短路, 保证不穿货架) -> 可视化标注到俯视图。

要点(踩过的坑): 障碍必须来自 sim 实际建出的 actor, 不能从 layout 猜 ——
未激活的货架不会建出, 不是障碍。
"""
import numpy as np


def _dedup_edges(nodes, adj):
    """同一对点只留一条边(取最短), 去掉自环。重复/平行边是度>2 的元凶之一。"""
    clean = {n: {} for n in nodes}
    for a, nbrs in adj.items():
        if a not in nodes:
            continue
        for b, w in nbrs:
            if b == a or b not in nodes:
                continue
            if b not in clean[a] or w < clean[a][b]:
                clean[a][b] = w
                clean[b][a] = w
    return {n: [(b, w) for b, w in d.items()] for n, d in clean.items()}


def contract_chains(nodes, adj, obstacles, keep=(), robot_radius=0.35):
    """链收缩: 直线通道只保留两端关节点, 中间度=2 的点全删。

    不靠共线容差, 只看度。1) 去重边/自环(度>2 元凶); 2) 关节点=度!=2 的点+keep;
    3) 每条"关节点-内部链-关节点"收缩成一条直连边(_seg_clear 验证不穿货架)。
    度=2 内部点必被吸收 -> 直线通道必只剩两端。Returns: (nodes, adj)。
    """
    keep = set(keep)
    adj = _dedup_edges(nodes, adj)
    fps = _footprints(obstacles, robot_radius)

    def degree(n):
        return len(adj.get(n, []))

    joints = {n for n in nodes if degree(n) != 2 or n in keep}
    new_adj = {n: [] for n in joints}
    seen_pairs = set()

    def add_edge(a, b):
        key = (min(a, b), max(a, b))
        if a == b or key in seen_pairs:
            return
        seen_pairs.add(key)
        d = float(np.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1]))
        new_adj[a].append((b, d))
        new_adj[b].append((a, d))

    for j in joints:
        for nb, _ in adj.get(j, []):
            prev, cur = j, nb
            while cur not in joints and degree(cur) == 2:
                nxt = [x for x, _ in adj[cur] if x != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            if cur in joints and _seg_clear(nodes[j], nodes[cur], fps):
                add_edge(j, cur)

    return {n: nodes[n] for n in joints}, new_adj


def prune_dead_ends(nodes, adj, keep=(), max_spur=0.5):
    """只剥短毛刺: 删度=1 且到其唯一邻居距离 < max_spur 的非 keep 节点。

    中轴在货架拐角长出的短毛刺(噪声)被剥掉; 通向真实走廊尽头的长端点(度=1
    但很长)保留 -> 不会像之前那样把真实路删没。循环, 直到无短毛刺。
    Returns: (nodes, adj)。
    """
    keep = set(keep)
    changed = True
    while changed:
        changed = False
        for n in list(nodes):
            if n in keep:
                continue
            nb = adj.get(n, [])
            if len(nb) != 1:
                continue
            b, _ = nb[0]
            d = float(np.hypot(nodes[n][0] - nodes[b][0], nodes[n][1] - nodes[b][1]))
            if d < max_spur:  # 短毛刺才删, 长端点保留
                adj[b] = [(x, w) for (x, w) in adj[b] if x != n]
                adj.pop(n, None)
                del nodes[n]
                changed = True
    return nodes, adj


def connect_components(nodes, adj, obstacles, robot_radius=0.35):
    """连通性修复: 若图分多个连通块, 用安全直线边把小块接回主块。

    medial axis 在开阔区可能断开 -> 几何上能连的块没连。这里对每个非主块, 找它到
    主块所有点对中最短且 _seg_clear(膨胀)安全的一条直边连上。连不上的块保持孤立
    (说明真被货架隔开)。返回 (adj, added) — added 为新增边数。
    """
    fps = _footprints(obstacles, robot_radius)

    def _components():
        seen, comps = set(), []
        for s in adj:
            if s in seen:
                continue
            c, st = [], [s]
            while st:
                u = st.pop()
                if u in seen:
                    continue
                seen.add(u); c.append(u)
                for v, _ in adj[u]:
                    if v not in seen:
                        st.append(v)
            comps.append(c)
        return comps

    added = 0
    while True:
        comps = _components()
        if len(comps) <= 1:
            break
        comps.sort(key=len, reverse=True)
        main = set(comps[0])
        merged = False
        for c in comps[1:]:
            best, bd = None, 1e18
            for n in c:
                for m in main:
                    d = float(np.hypot(nodes[n][0] - nodes[m][0],
                                       nodes[n][1] - nodes[m][1]))
                    if d < bd and _seg_clear(nodes[n], nodes[m], fps):
                        bd, best = d, (n, m)
            if best is not None:
                n, m = best
                adj[n].append((m, bd)); adj[m].append((n, bd))
                added += 1
                merged = True
                break          # 重算连通块再找下一个
        if not merged:
            break              # 剩余块都被货架真隔开, 无安全边可连
    return adj, added


def prune_unsafe_edges(nodes, adj, obstacles, robot_radius=0.35):
    """最终安全兜底: 删掉所有穿膨胀货架的边(不管哪个阶段引入)。

    硬保证 _seg_clear: 任何切进膨胀货架的边都移除 -> 机器人不会贴/撞货架。
    返回 (adj, removed) — removed 为被删边数, >0 说明上游有引入坏边的 bug。
    """
    fps = _footprints(obstacles, robot_radius)
    removed = 0
    for a in list(adj):
        kept = []
        for b, w in adj[a]:
            if b in nodes and _seg_clear(nodes[a], nodes[b], fps):
                kept.append((b, w))
            else:
                removed += 1
        adj[a] = kept
    return adj, removed // 2


def merge_close_joints(nodes, adj, keep=(), radius=0.3, obstacles=None,
                       robot_radius=0.35):
    """合并 radius 内的近距离关节点对: 链收缩后仍重合的关节点塌成一个。

    contract_chains 只吸收度=2 内部点; 若两个关节点(度!=2)彼此很近会双双保留 ->
    残留近邻重合点。这里把 < radius 的关节点对合并: 非 keep 点并入对端, 边重连去重。
    keep 点不被并入别处(作簇心吸收近邻)。给 obstacles 时, 合并后产生的新边须过
    _seg_clear(膨胀 robot_radius)验证, 穿货架的边丢弃 -> 不会贴/撞货架。
    循环直到无可合并对。Returns: (nodes, adj)。
    """
    keep = set(keep)
    r2 = radius * radius
    fps = _footprints(obstacles, robot_radius) if obstacles is not None else None
    changed = True
    while changed:
        changed = False
        names = list(nodes)
        for i in range(len(names)):
            a = names[i]
            if a not in nodes:
                continue
            for j in range(i + 1, len(names)):
                b = names[j]
                if b not in nodes or b == a:
                    continue
                dx = nodes[a][0] - nodes[b][0]
                dy = nodes[a][1] - nodes[b][1]
                if dx * dx + dy * dy > r2:
                    continue
                # 决定保留谁: keep 点优先留; 否则留 a, 把 b 并入 a
                if b in keep and a not in keep:
                    a, b = b, a
                # 把 b 的邻居改接到 a(去自环, 验证新边不穿膨胀货架), 再删 b
                for nb, w in adj.get(b, []):
                    if nb == a or nb == b:
                        continue
                    adj[nb] = [(x, ww) for (x, ww) in adj.get(nb, []) if x != b]
                    if fps is not None and not _seg_clear(nodes[a], nodes[nb], fps):
                        continue  # 新边会切进膨胀货架 -> 不连
                    d = float(np.hypot(nodes[a][0] - nodes[nb][0],
                                       nodes[a][1] - nodes[nb][1]))
                    if not any(x == nb for x, _ in adj[a]):
                        adj[a].append((nb, d))
                    if not any(x == a for x, _ in adj.get(nb, [])):
                        adj[nb].append((a, d))
                adj[a] = [(x, w) for (x, w) in adj[a] if x != b]
                adj.pop(b, None)
                del nodes[b]
                changed = True
                break
            if changed:
                break
    return nodes, adj


def cleanup_graph(nodes, adj, obstacles, keep=(), robot_radius=0.35):
    """链收缩精简: 去重边 -> 按度找关节点 -> 直线通道收缩成两端点。

    根治"直线通道残留大量结点": contract_chains 只看度, 度=2 内部点必被吸收。
    收缩后再剥短毛刺(中轴拐角噪声), 得到只剩路口+端点的干净图。
    最后合并近距离关节点对(链收缩吸收不掉的近邻重合点), 彻底去冗余。
    """
    keep = set(keep)
    nodes, adj = contract_chains(nodes, adj, obstacles, keep=keep,
                                 robot_radius=robot_radius)
    prune_dead_ends(nodes, adj, keep=keep)
    # 剥毛刺可能让某些关节点退化为度=2, 再收缩一次彻底干净
    nodes, adj = contract_chains(nodes, adj, obstacles, keep=keep,
                                 robot_radius=robot_radius)
    # 合并近距离关节点对 -> 再收缩一次(合并可能让某点重回度=2)
    merge_close_joints(nodes, adj, keep=keep)
    nodes, adj = contract_chains(nodes, adj, obstacles, keep=keep,
                                 robot_radius=robot_radius)
    return nodes, adj


def corridor_centerlines(obstacles, scene_size, robot_radius=0.35, margin=0.4,
                         min_gap=0.7, key_points=None):
    """通道中线法(纯几何, 不用中轴): 轴对齐货架 -> 行列间隙中线 -> 交点=路口。

    货架投影到 x 轴, 列间空隙中心 = 竖直通道中线; 投影到 y 轴, 行间空隙中心 =
    水平通道中线。横线×竖线交点 = 路口(每个十字仅一点)。直通道 = 两端点一条直边,
    无毛刺、无冗余点。返回 (nodes{name:[x,y]}, vx竖线x表, hy横线y表)。
    """
    sx, sy = scene_size
    xs_occ = [(cx - hl - robot_radius, cx + hl + robot_radius)
              for cx, cy, hl, hw in obstacles]
    ys_occ = [(cy - hw - robot_radius, cy + hw + robot_radius)
              for cx, cy, hl, hw in obstacles]
    vx = _free_bands(xs_occ, margin, sx - margin, min_gap)  # 竖直通道中线 x
    hy = _free_bands(ys_occ, margin, sy - margin, min_gap)  # 水平通道中线 y
    fps = _footprints(obstacles, robot_radius)
    nodes, idx = {}, 0
    for x in vx:
        for y in hy:
            if not _inside(x, y, fps):  # 交点须在自由空间
                nodes[f"J{idx}"] = [round(float(x), 2), round(float(y), 2)]
                idx += 1
    if key_points:
        for name, (x, y) in key_points.items():
            nodes[name] = [round(float(x), 2), round(float(y), 2)]
    return nodes, vx, hy


def edges_on_centerlines(nodes, vx, hy, obstacles, robot_radius=0.35, tol=0.15):
    """沿通道中线连边: 同一条竖线/横线上相邻路口直连(验证不穿货架)。

    直通道 -> 一条边连两端; 不产生跨点冗余边。Returns: 邻接表。
    """
    fps = _footprints(obstacles, robot_radius)
    adj = {n: [] for n in nodes}

    def _link(seq):
        seq = sorted(seq, key=lambda t: t[1])
        for i in range(len(seq) - 1):
            a, b = seq[i][0], seq[i + 1][0]
            if _seg_clear(nodes[a], nodes[b], fps):
                d = float(np.hypot(nodes[a][0] - nodes[b][0],
                                   nodes[a][1] - nodes[b][1]))
                adj[a].append((b, d))
                adj[b].append((a, d))

    for x in vx:  # 每条竖线: 按 y 排序相邻连
        _link([(n, nodes[n][1]) for n in nodes if abs(nodes[n][0] - x) < tol])
    for y in hy:  # 每条横线: 按 x 排序相邻连
        _link([(n, nodes[n][0]) for n in nodes if abs(nodes[n][1] - y) < tol])
    return adj


def merge_close_nodes(nodes, radius=0.6, keep=()):
    """把半径内成簇的节点合并成一个(取质心)。

    解决中轴在十字/丁字路口分叉成一簇相邻点的问题 -> 每个路口塌成单点。
    keep 中的节点(起点/approach)不被并入别处, 但可吸收附近点。
    Returns: 合并后的 {name:[x,y]}。
    """
    keep = set(keep)
    items = list(nodes.items())
    used = set()
    merged = {}
    r2 = radius * radius
    # 优先以 keep 节点为簇心, 再处理其余
    order = [n for n in nodes if n in keep] + [n for n in nodes if n not in keep]
    for n in order:
        if n in used:
            continue
        cx, cy = nodes[n]
        cluster = [n]
        for m, (mx, my) in items:
            if m == n or m in used or m in keep:
                continue
            if (mx - cx) ** 2 + (my - cy) ** 2 <= r2:
                cluster.append(m)
        for m in cluster:
            used.add(m)
        if n in keep:
            merged[n] = nodes[n]  # keep 点坐标不动
        else:
            xs = [nodes[c][0] for c in cluster]
            ys = [nodes[c][1] for c in cluster]
            merged[n] = [round(sum(xs) / len(xs), 2), round(sum(ys) / len(ys), 2)]
    return merged


def obstacles_from_layout(scene_dir, env=None):
    """从 layout 读全部货架(active+inactive)当障碍 —— 含 sim 未 build 的仓库货架。

    踩坑修正: sim 只 build active 货架, 仓库整排(layout 有、sim 无)读不到 ->
    A* 把仓库当空地穿过去。改从 layout 取全部货架, 用 l/w/orientation 算 footprint。
    commercial 占位坐标为 (0,0), 用 sim 真实 active 货架位置补上。
    Returns: list of (cx, cy, half_l, half_w)。
    """
    from generate_topdown_map import load_layout
    ld = load_layout(scene_dir)["layout_data"]

    def _half(l, w, orient):
        return (w / 2.0, l / 2.0) if orient == "vertical" else (l / 2.0, w / 2.0)

    obs = []
    for grp in ("inactive_shelvings", "active_shelvings",
                "inactive_wall_shelvings", "active_wall_shelvings"):
        for s in ld.get(grp, []):
            x, y = float(s.get("x", 0)), float(s.get("y", 0))
            l, w = float(s.get("l", 1.55)), float(s.get("w", 0.55))
            hl, hw = _half(l, w, s.get("orientation", "horizontal"))
            if abs(x) < 1e-6 and abs(y) < 1e-6:
                continue  # (0,0) 占位, 下面用 sim 真实位置补
            obs.append((round(x, 2), round(y, 2), round(hl, 3), round(hw, 3)))

    # 用 sim 真实 active 货架位置补 (0,0) 占位
    if env is not None:
        u = env.unwrapped if hasattr(env, "unwrapped") else env
        for k, a in u.actors.get("fixtures", {}).get("shelves", {}).items():
            if "active_" in k and "inactive" not in k:
                p = np.asarray(a.pose.sp.p)
                px, py = round(float(p[0]), 2), round(float(p[1]), 2)
                if any(abs(px - o[0]) < 0.3 and abs(py - o[1]) < 0.3 for o in obs):
                    continue  # 已在 layout 障碍里
                obs.append((px, py, 0.78, 0.28))
    return obs


def obstacles_from_sim(env):
    """从 sim 实际建出的 shelf actor 提取障碍 footprint(用真实碰撞网格包围盒)。

    直接读 actor 碰撞网格的 AABB, 不写死尺寸 —— 写死会导致膨胀错误、A* 贴边穿货架。
    Returns: list of (cx, cy, half_l, half_w) 轴对齐包围盒(米)。
    """
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    raw = u.actors.get("fixtures", {}).get("shelves", {})
    obs = []
    for k, a in raw.items():
        p = np.asarray(a.pose.sp.p)
        hl, hw = None, None
        for mname in ("get_collision_meshes", "get_first_collision_mesh"):
            if hasattr(a, mname):
                try:
                    r = getattr(a, mname)()
                    mesh = r[0] if isinstance(r, list) else r
                    lo, hi = mesh.bounds
                    hl = float(hi[0] - lo[0]) / 2.0
                    hw = float(hi[1] - lo[1]) / 2.0
                    break
                except Exception:
                    pass
        if hl is None:  # 拿不到网格时退回典型金属货架尺寸 1.55x0.55
            m = a.pose.sp.to_transformation_matrix()
            yaw = np.arctan2(m[1, 0], m[0, 0])
            c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
            hl = 0.775 * c + 0.275 * s
            hw = 0.775 * s + 0.275 * c
        obs.append((round(float(p[0]), 2), round(float(p[1]), 2),
                    round(hl, 3), round(hw, 3)))
    return obs


def _footprints(obs, r):
    return [(cx - hl - r, cx + hl + r, cy - hw - r, cy + hw + r)
            for cx, cy, hl, hw in obs]


def _inside(x, y, fps):
    for x0, x1, y0, y1 in fps:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _seg_clear(p, q, fps, step=0.08):
    """线段 p->q 是否完全不穿任何(膨胀后)障碍。"""
    d = float(np.hypot(q[0] - p[0], q[1] - p[1]))
    n = max(2, int(d / step))
    for t in np.linspace(0.0, 1.0, n):
        if _inside(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, fps):
            return False
    return True


def _free_bands(occupied, lo, hi, min_gap):
    """[lo,hi] 上去掉 occupied 区间后, 每条自由带的中心线坐标。"""
    centers, cursor = [], lo
    for a, b in sorted(occupied):
        if a - cursor >= min_gap:
            centers.append(round((cursor + a) / 2, 2))
        cursor = max(cursor, b)
    if hi - cursor >= min_gap:
        centers.append(round((cursor + hi) / 2, 2))
    return centers


def corridor_intersections(obstacles, scene_size, robot_radius=0.35,
                           margin=0.5, grid=1.0, key_points=None):
    """全场景自由空间路口网格(纯几何, 无搜索算法)。

    在整个场景(含未激活区)铺 grid 间距网格, 落在自由空间(不在任何膨胀障碍
    内)的格点即为可用路口。VLM 输出像素路径 -> 转真实坐标 -> nearest_node 匹配
    到最近路口; 故路口越密匹配越准, 数量不构成 VLM 负担。

    Args:
        obstacles: [(cx,cy,half_l,half_w), ...] 来自 obstacles_from_sim。
        scene_size: [sx, sy]。
        robot_radius: 障碍膨胀半径(米)。
        margin: 距墙内缩(米)。
        grid: 网格间距(米), 越小路口越密。
        key_points: {name:[x,y]} 额外纳入的命名点(休息区、货架接近点)。

    Returns: {name: [x,y]} 路口及关键点坐标表。
    """
    sx, sy = scene_size
    fps = _footprints(obstacles, robot_radius)
    nodes, idx = {}, 0
    xs = np.arange(margin, sx - margin + 1e-6, grid)
    ys = np.arange(margin, sy - margin + 1e-6, grid)
    for xi in xs:
        for yi in ys:
            x, y = round(float(xi), 2), round(float(yi), 2)
            if not _inside(x, y, fps):
                nodes[f"J{idx}"] = [x, y]
                idx += 1
    if key_points:
        for name, (x, y) in key_points.items():
            nodes[name] = [round(float(x), 2), round(float(y), 2)]
    return nodes


def nearest_node(nodes, xy):
    """把任意坐标匹配到最近的路口/关键点, 返回 (name, [x,y])。"""
    best, bd = None, 1e18
    for name, (nx, ny) in nodes.items():
        d = (nx - xy[0]) ** 2 + (ny - xy[1]) ** 2
        if d < bd:
            bd, best = d, name
    return best, nodes[best]


def _trace_skeleton_edges(skel, junc, xs, ys, nodes, sample_step, fps):
    """沿骨架追踪相邻路口连边: 从每个路口走 度=2 的中线链到下一个路口。

    像高德路网: 两路口间的中线链尽量收缩成 ONE 直边, 但 *弯的* 通道(L 形拐角是
    度=2 不算路口)若直连会切角、贴近膨胀货架 -> 用 _seg_clear 验证, 失败处保留
    拐点节点, 拆成多段都不穿膨胀障碍的直边。返回世界坐标边列表 [((x0,y0),(x1,y1)),..]。
    """
    import numpy as _np
    ny, nx = skel.shape
    jset = {(int(r), int(c)) for r, c in zip(*_np.where(junc))}

    def _node_at(r, c):
        wx, wy = float(xs[c]), float(ys[r])
        best, bd = None, (sample_step * 0.9) ** 2
        for nm, (px, py) in nodes.items():
            d = (px - wx) ** 2 + (py - wy) ** 2
            if d < bd:
                bd, best = d, nm
        return best

    def _nbrs(r, c):
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < ny and 0 <= cc < nx and skel[rr, cc]:
                    out.append((rr, cc))
        return out

    bend = [0]

    def _add_bend(wx, wy):
        nm = f"B{bend[0]}"; bend[0] += 1
        nodes[nm] = [round(wx, 2), round(wy, 2)]
        return nm

    edges, seen = [], set()
    for jr, jc in jset:
        a = _node_at(jr, jc)
        if a is None:
            continue
        for nr, nc in _nbrs(jr, jc):
            prev, cur = (jr, jc), (nr, nc)
            chain = [(jr, jc), (nr, nc)]
            steps = 0
            while cur not in jset and steps < 100000:
                nxt = [p for p in _nbrs(*cur) if p != prev]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
                chain.append(cur); steps += 1
            if cur not in jset:
                continue
            b = _node_at(*cur)
            if b is None or b == a:
                continue
            mid = chain[len(chain) // 2]
            key = (min(a, b), max(a, b), mid)   # mid 区分平行通道, 也去重反向走
            if key in seen:
                continue
            seen.add(key)
            # 链像素 -> 世界点(两端用节点坐标), 贪心切成都过 _seg_clear 的直边
            wpts = [(float(xs[c]), float(ys[r])) for (r, c) in chain]
            wpts[0], wpts[-1] = tuple(nodes[a]), tuple(nodes[b])
            kept = [0]; i = 0
            while i < len(wpts) - 1:
                j = len(wpts) - 1
                while j > i + 1 and not _seg_clear(wpts[i], wpts[j], fps):
                    j -= 1
                kept.append(j); i = j
            seq = [a] + [_add_bend(*wpts[k]) for k in kept[1:-1]] + [b]
            for u, v in zip(seq, seq[1:]):
                edges.append((nodes[u], nodes[v]))
    return edges


def medial_axis_nodes(obstacles, scene_size, robot_radius=0.35, margin=0.4,
                      res=0.05, sample_step=0.6, key_points=None,
                      return_skeleton=False):
    """中轴骨架法: 把通道当"路", 中线交点/端点当"路口"(像高德路网)。

    自由空间栅格化 -> medial_axis 抽通道中线骨架 -> 在骨架上找分叉点/端点(=路口)
    并沿骨架等距采样补点 -> 这些点即路网节点。比网格法稀疏、贴合通道中央。

    Args:
        obstacles, scene_size, robot_radius, margin: 同前。
        res: 栅格分辨率(米/像素)。
        sample_step: 沿骨架补点的间距(米)。
        key_points: {name:[x,y]} 额外纳入(休息区等)。
        return_skeleton: True 时额外返回 (骨架世界坐标点列表, 是否分叉点)。
    Returns: {name:[x,y]}  或  ({name:[x,y]}, skel_xy) 当 return_skeleton。
    """
    from skimage.morphology import medial_axis
    sx, sy = scene_size
    nx, ny = int(sx / res), int(sy / res)
    fps = _footprints(obstacles, robot_radius)
    free = np.ones((ny, nx), dtype=bool)
    xs = (np.arange(nx) + 0.5) * res
    ys = (np.arange(ny) + 0.5) * res
    for x0, x1, y0, y1 in fps:
        ix0, ix1 = max(0, int(x0 / res)), min(nx, int(np.ceil(x1 / res)))
        iy0, iy1 = max(0, int(y0 / res)), min(ny, int(np.ceil(y1 / res)))
        free[iy0:iy1, ix0:ix1] = False
    m = int(margin / res)
    if m > 0:
        free[:m, :] = free[-m:, :] = free[:, :m] = free[:, -m:] = False

    skel = medial_axis(free)
    # 邻居数: ==1 端点, >=3 分叉点 -> 都是路口
    from scipy.ndimage import convolve
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    deg = convolve(skel.astype(int), k, mode="constant") * skel
    junc = (skel & ((deg == 1) | (deg >= 3)))

    nodes, idx, placed = {}, 0, []

    def _add(wx, wy):
        nonlocal idx
        for px, py in placed:
            if (px - wx) ** 2 + (py - wy) ** 2 < (sample_step * 0.7) ** 2:
                return
        nodes[f"J{idx}"] = [round(wx, 2), round(wy, 2)]
        placed.append((wx, wy)); idx += 1

    jy, jx = np.where(junc)
    for r, c in zip(jy, jx):
        _add(float(xs[c]), float(ys[r]))
    # 像高德路网: 只保留分叉点(度>=3)和端点(度=1)作为路口, 直通道中间不撒点。
    # 边由沿骨架追踪相邻路口直连得到 -> 一条通道一条边, 无冗余中间点。
    edges = _trace_skeleton_edges(skel, junc, xs, ys, nodes, sample_step, fps)

    if key_points:
        for name, (x, y) in key_points.items():
            nodes[name] = [round(float(x), 2), round(float(y), 2)]

    if return_skeleton:
        syk, sxk = np.where(skel)
        skel_xy = [[float(xs[c]), float(ys[r])] for r, c in zip(syk, sxk)]
        return nodes, edges, skel_xy
    return nodes, edges


def attach_point(nodes, adj, obstacles, name, xy, robot_radius=0.35,
                 max_connect=2.5, max_edges=4):
    """把一个目标点(如货架 approach 位)接入路网: 加为节点 + 连边到可见的中线节点。

    approach 点天生贴货架(在货架前 1.4m 取放货位), 用满膨胀(robot_radius)验证连线常
    过严而连不上 -> 孤立点, A* 到不了。故采用 *分级回退*: 从 robot_radius 起逐级放宽膨胀
    (0.45/0.3/0.1/0), 第一个能连上 >=1 条边的档位即采用。若 0 膨胀(实墙)都连不上, 兜底
    连最近的节点(保证连通; 该段贴障由 NavDP 局部避障兜)。
    max_edges 限制每档最多连几个最近可见点(默认4, 设1则只连最近一个)。
    Returns: True(至少连上一条边, 含兜底)。
    """
    nodes[name] = [round(float(xy[0]), 2), round(float(xy[1]), 2)]
    adj.setdefault(name, [])
    levels = [r for r in (robot_radius, 0.45, 0.3, 0.1, 0.0) if r <= robot_radius]
    if 0.0 not in levels:
        levels.append(0.0)
    for r in levels:
        fps = _footprints(obstacles, r)
        cand = []
        for other, (ox, oy) in nodes.items():
            if other == name:
                continue
            d = float(np.hypot(ox - xy[0], oy - xy[1]))
            if d <= max_connect and _seg_clear(xy, [ox, oy], fps):
                cand.append((d, other))
        if cand:
            cand.sort()
            for d, other in cand[:max_edges]:
                adj[name].append((other, d))
                adj[other].append((name, d))
            return True
    # 兜底: 0 膨胀都连不上(approach 与所有路点间都有实墙) -> 连最近节点保证连通
    nearest = min(((float(np.hypot(ox - xy[0], oy - xy[1])), o)
                   for o, (ox, oy) in nodes.items() if o != name), default=None)
    if nearest is not None:
        d, other = nearest
        adj[name].append((other, d))
        adj[other].append((name, d))
    return len(adj[name]) > 0


def build_edges(nodes, obstacles, robot_radius=0.35, max_len=2.2, k=4):
    """连接互相可见(直线不穿膨胀障碍)的路口对; 每点只保留最近的 k 条边。

    max_len 限制只连邻近路口; k-NN 限制每点边数 -> 避免转角处点簇两两全连成密扇形,
    图稀疏干净且仍连通。每条边都验证不穿货架 -> A* 路径整体避障。
    Returns: {name: [(neighbor, dist), ...]} 邻接表。
    """
    fps = _footprints(obstacles, robot_radius)
    names = list(nodes)
    # 先按距离+可见性收集每点的候选边
    cand = {n: [] for n in names}
    for i in range(len(names)):
        a = names[i]
        pa = nodes[a]
        for j in range(i + 1, len(names)):
            b = names[j]
            pb = nodes[b]
            d = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
            if d > max_len:
                continue
            if _seg_clear(pa, pb, fps):
                cand[a].append((d, b))
                cand[b].append((d, a))
    # 每点只取最近 k 条(对称去重 -> 用 set)
    edge_set = set()
    for a in names:
        for d, b in sorted(cand[a])[:k]:
            edge_set.add((min(a, b), max(a, b), round(d, 3)))
    adj = {n: [] for n in names}
    for a, b, d in edge_set:
        adj[a].append((b, d))
        adj[b].append((a, d))
    return adj


def astar(nodes, adj, start, goal):
    """A* 在路口图上求 start->goal 最短路(节点名列表)。无路返回 None。

    所有边已验证不穿货架, 故返回路径整体不穿货架 = 全局避障。
    """
    import heapq

    def h(n):
        a, b = nodes[n], nodes[goal]
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    openq = [(h(start), 0.0, start, [start])]
    best = {start: 0.0}
    while openq:
        _, g, cur, path = heapq.heappop(openq)
        if cur == goal:
            return path
        for nxt, w in adj.get(cur, []):
            ng = g + w
            if ng < best.get(nxt, 1e18):
                best[nxt] = ng
                heapq.heappush(openq, (ng + h(nxt), ng, nxt, path + [nxt]))
    return None


def render_clean_topdown(obstacles, scene_size, out_path, robot_xy=None,
                         px_per_m=60):
    """渲染坐标映射固定的俯视图给 VLM(无 tight bbox, 像素↔世界严格线性)。

    图覆盖 [0,sx]x[0,sy], 原点(0,0)在左下。返回 (W, H) 像素尺寸, 供 pixel<->world。
    货架画成灰块, 机器人画成红点; 不标路口(VLM 自行判断通道)。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sx, sy = scene_size
    W, H = int(sx * px_per_m), int(sy * px_per_m)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])  # 铺满画布, 无边距
    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)
    ax.set_facecolor("#f0ece4")
    for cx, cy, hl, hw in obstacles:
        ax.add_patch(Rectangle((cx - hl, cy - hw), 2 * hl, 2 * hw,
                               facecolor="#777", edgecolor="#333", linewidth=1))
    if robot_xy is not None:
        ax.plot(robot_xy[0], robot_xy[1], "o", ms=14, color="red")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return W, H


def pixel_to_world(px, py, img_wh, scene_size):
    """图像像素(原点左上, y 向下) -> 世界坐标(原点左下, y 向上)。"""
    W, H = img_wh
    sx, sy = scene_size
    x = px / W * sx
    y = (1.0 - py / H) * sy
    return [round(float(x), 2), round(float(y), 2)]


def render_graph(obstacles, scene_size, nodes, adj, out_path, robot_xy=None,
                 path=None, px_per_m=60, show_labels=True, skeleton=None,
                 approach_pts=None):
    """把路口图(点+边)画到俯视图上, 供人工检查图是否正确。

    灰块=货架, 红星=机器人, 蓝点=路口, 青线=验证过的边, 橙线=A*路径(若给),
    浅灰点=中轴线骨架(skeleton), 品红方块=货架 approach 操作位(approach_pts)。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sx, sy = scene_size
    W, H = sx * px_per_m / 100, sy * px_per_m / 100
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)
    ax.set_aspect("equal")
    ax.set_facecolor("#f0ece4")
    for cx, cy, hl, hw in obstacles:
        ax.add_patch(Rectangle((cx - hl, cy - hw), 2 * hl, 2 * hw,
                               facecolor="#888", edgecolor="#333", linewidth=1, zorder=1))
    # 中轴线骨架(通道中线本身)
    if skeleton:
        ax.plot([p[0] for p in skeleton], [p[1] for p in skeleton], ".",
                color="#bbb", ms=1.2, zorder=1.5)
    # 边(去重: 只画 a<b)
    drawn = set()
    for a, nbrs in adj.items():
        for b, _ in nbrs:
            key = tuple(sorted([a, b]))
            if key in drawn:
                continue
            drawn.add(key)
            pa, pb = nodes[a], nodes[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], "-", color="#5a9",
                    lw=0.8, alpha=0.6, zorder=2)
    # 路口点
    for name, (x, y) in nodes.items():
        ax.plot(x, y, "o", ms=5, color="#06c", zorder=3)
        if show_labels:
            ax.annotate(name, (x, y), fontsize=5, color="#024", zorder=4)
    # 货架 approach 操作位
    if approach_pts:
        for name, (x, y) in approach_pts.items():
            ax.plot(x, y, "s", ms=9, color="magenta", zorder=6)
            ax.annotate(name, (x, y), fontsize=7, color="purple", zorder=7)
    # A* 路径
    if path:
        pts = [nodes[n] for n in path]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-",
                color="orange", lw=3, zorder=5)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o",
                color="orange", ms=8, zorder=6)
    if robot_xy is not None:
        ax.plot(robot_xy[0], robot_xy[1], "*", ms=20, color="red", zorder=7)
    n_edges = sum(len(v) for v in adj.values()) // 2
    ax.set_title(f"{len(nodes)} junctions, {n_edges} edges")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return len(nodes), n_edges

