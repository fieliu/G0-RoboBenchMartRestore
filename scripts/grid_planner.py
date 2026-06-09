"""栅格占用图 + 梯度膨胀代价地图 + A* 全局规划(拓扑图之外的另一种 planner)。

与 corridor_graph(拓扑图)互不影响; 两者障碍同源(obstacles_from_sim/layout),
输出同构(可喂同一套 NavDP 与可视化)。本模块只负责"算路"。

思路(对应面试口径):
  1. 障碍栅格化: 10cm/格, 障碍格标占用。
  2. 梯度膨胀代价地图(costmap): 距每格最近障碍的距离 d ->
       d < robot_radius(0.1m): 致命/内切, 不可行;
       robot_radius <= d < inflation_radius(1.5m): 代价 exp 衰减(贴障高、远离低);
       d >= inflation_radius: 代价 0。
     A* 代价 = 移动距离 + 障碍代价 -> 自动走通道正中(梯度让中线代价最低)。
  3. 栅格 A*(8 邻接, 对角 *sqrt2), h=欧氏距离。
  4. 目标点豁免: 贴货架的 approach 点可能落在内切层 -> 局部豁免保证可达,
     最后一小段贴障交给 NavDP 局部避障兜。
  5. 路径抽稀: 稠密格路径 -> 拐点路点(Douglas-Peucker), 喂 NavDP。
"""
import numpy as np


class GridPlanner:
    """占用栅格 + 梯度膨胀 + A*。坐标系: 世界 (x,y) 米, 原点左下; 栅格 (row=y, col=x)。"""

    def __init__(self, obstacles, scene_size, res=0.1, robot_radius=0.1,
                 inflation_radius=1.5, margin=0.4, lethal_cost=1e6,
                 inflation_gain=8.0, inflation_weight=30.0):
        """
        Args:
            obstacles: [(cx,cy,half_l,half_w), ...] 来自 obstacles_from_sim/layout。
            scene_size: [sx, sy] 场景尺寸(米)。
            res: 栅格分辨率(米/格), 默认 0.1m。
            robot_radius: 机器人半径(米), 内切禁区半径, 默认 0.1m。
            inflation_radius: 梯度膨胀半径(米), 此距离外代价为 0, 默认 1.5m。
            margin: 距场景边界内缩(米), 当墙处理。
            lethal_cost: 致命/内切层代价(视作不可行)。
            inflation_gain: 指数衰减速率 k, 越大衰减越快。
            inflation_weight: 梯度层代价幅度(乘到 exp 上), 越大越倾向走中间。
        """
        self.res = res
        self.robot_radius = robot_radius
        self.inflation_radius = inflation_radius
        self.lethal_cost = lethal_cost
        self.inflation_gain = inflation_gain
        self.inflation_weight = inflation_weight
        self.sx, self.sy = float(scene_size[0]), float(scene_size[1])
        self.nx = int(np.ceil(self.sx / res))
        self.ny = int(np.ceil(self.sy / res))
        self._build_costmap(obstacles, margin)

    # ── 坐标转换 ───────────────────────────────────────────────
    def world_to_grid(self, x, y):
        c = int(x / self.res)
        r = int(y / self.res)
        return min(max(r, 0), self.ny - 1), min(max(c, 0), self.nx - 1)

    def grid_to_world(self, r, c):
        return [round((c + 0.5) * self.res, 3), round((r + 0.5) * self.res, 3)]

    # ── 代价地图 ───────────────────────────────────────────────
    def _build_costmap(self, obstacles, margin):
        """占用栅格 -> 距离变换 -> 梯度膨胀代价地图。"""
        occ = np.zeros((self.ny, self.nx), dtype=bool)
        for cx, cy, hl, hw in obstacles:
            r0, c0 = self.world_to_grid(cx - hl, cy - hw)
            r1, c1 = self.world_to_grid(cx + hl, cy + hw)
            occ[min(r0, r1):max(r0, r1) + 1, min(c0, c1):max(c0, c1) + 1] = True
        # 边界 margin 当墙
        m = int(margin / self.res)
        if m > 0:
            occ[:m, :] = occ[-m:, :] = occ[:, :m] = occ[:, -m:] = True

        # 到最近障碍的距离(米): 用 EDT, 距离 = 自由格到最近占用格
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(~occ) * self.res  # 每格到最近 True(障碍)的距离

        cost = np.zeros((self.ny, self.nx), dtype=np.float64)
        # 致命/内切层: 障碍本体 + 距障碍 < robot_radius
        lethal = occ | (dist < self.robot_radius)
        cost[lethal] = self.lethal_cost
        # 梯度衰减层: robot_radius <= d < inflation_radius
        grad = (~lethal) & (dist < self.inflation_radius)
        cost[grad] = self.inflation_weight * np.exp(
            -self.inflation_gain * (dist[grad] - self.robot_radius))
        # d >= inflation_radius: cost = 0(已初始化)
        self.occ = occ
        self.dist = dist
        self.cost = cost
        self.lethal = lethal

    def _free(self, r, c):
        return not self.lethal[r, c]

    def _exempt_goal(self, r, c, radius_cells=None):
        """目标豁免: 贴货架的 approach 点若落在内切层, 在其周围小邻域降代价保证可达。

        只把目标点本身及紧邻致命格设为'高代价但可行', 不破坏全局避障; 最后贴障
        一小段由 NavDP 局部避障兜。返回 (r,c) 调整到邻域内代价最低的可行格。
        """
        if self._free(r, c):
            return r, c
        # 在小窗口内找最近的可行格作为实际目标
        rad = radius_cells or int(np.ceil(self.robot_radius / self.res)) + 2
        best, bd = None, 1e18
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.ny and 0 <= cc < self.nx and self._free(rr, cc):
                    d = dr * dr + dc * dc
                    if d < bd:
                        bd, best = d, (rr, cc)
        if best is not None:
            return best
        # 邻域内全是致命(目标深陷障碍): 临时把目标格设为可行(高代价), 让 A* 至少能到
        self.cost[r, c] = self.lethal_cost * 0.5
        self.lethal[r, c] = False
        return r, c

    # ── A* ────────────────────────────────────────────────────
    def astar_grid(self, start_world, goal_world):
        """栅格 A*: 8 邻接, 代价=移动距离+障碍代价, h=欧氏距离。返回格路径 [(r,c),...] 或 None。"""
        import heapq
        sr, sc = self.world_to_grid(*start_world)
        gr, gc = self.world_to_grid(*goal_world)
        # 起点/终点若落致命层 -> 豁免到最近可行格
        sr, sc = self._exempt_goal(sr, sc)
        gr, gc = self._exempt_goal(gr, gc)
        goal = (gr, gc)

        def h(r, c):
            return float(np.hypot(r - gr, c - gc))

        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        openq = [(h(sr, sc), 0.0, (sr, sc))]
        g_score = {(sr, sc): 0.0}
        came = {}
        while openq:
            _, g, cur = heapq.heappop(openq)
            if cur == goal:
                return self._reconstruct(came, cur)
            cr, cc = cur
            if g > g_score.get(cur, 1e18):
                continue
            for dr, dc, step in nbrs:
                rr, cc2 = cr + dr, cc + dc
                if not (0 <= rr < self.ny and 0 <= cc2 < self.nx):
                    continue
                if self.lethal[rr, cc2]:
                    continue
                # 对角穿障防漏(两正交格任一致命则禁止对角)
                if dr != 0 and dc != 0:
                    if self.lethal[cr, cc2] or self.lethal[rr, cc]:
                        continue
                ng = g + step + self.cost[rr, cc2] * self.res
                if ng < g_score.get((rr, cc2), 1e18):
                    g_score[(rr, cc2)] = ng
                    came[(rr, cc2)] = cur
                    heapq.heappush(openq, (ng + h(rr, cc2), ng, (rr, cc2)))
        return None

    @staticmethod
    def _reconstruct(came, cur):
        path = [cur]
        while cur in came:
            cur = came[cur]
            path.append(cur)
        return path[::-1]

    # ── 路径抽稀 ───────────────────────────────────────────────
    def simplify(self, grid_path, epsilon_m=0.15):
        """Douglas-Peucker 抽稀: 稠密格路径 -> 少量拐点路点(世界坐标列表)。"""
        if not grid_path:
            return []
        pts = [self.grid_to_world(r, c) for r, c in grid_path]
        kept = self._dp(pts, epsilon_m)
        return kept

    def _dp(self, pts, eps):
        if len(pts) < 3:
            return pts[:]
        # 找离首尾连线最远的点
        a, b = np.array(pts[0]), np.array(pts[-1])
        ab = b - a
        ab_norm = np.hypot(*ab) + 1e-9
        dmax, idx = 0.0, 0
        for i in range(1, len(pts) - 1):
            p = np.array(pts[i])
            d = abs(np.cross(ab, p - a)) / ab_norm
            if d > dmax:
                dmax, idx = d, i
        if dmax > eps:
            left = self._dp(pts[:idx + 1], eps)
            right = self._dp(pts[idx:], eps)
            return left[:-1] + right
        return [pts[0], pts[-1]]

    def plan(self, start_world, goal_world, epsilon_m=0.15):
        """端到端: A* 算格路径 -> 抽稀成世界路点序列。无解返回 None。"""
        gp = self.astar_grid(start_world, goal_world)
        if gp is None:
            return None
        return self.simplify(gp, epsilon_m)
