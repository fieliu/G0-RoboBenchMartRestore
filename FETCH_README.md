# Fetch 跨本体微调 + 分层部署

把预训练的 **G0Plus**(R1Lite, 26-DoF) 通过 LoRA 跨本体微调到 **Fetch**(15-DoF), 并用一个
**VLM 规划器 + VLA/导航执行器** 的分层架构部署到商超补货任务。

数据来自 [RoboBenchMart](https://github.com/) 的 Fetch (`ds_fetch_basket`) 运动规划轨迹。

---

## 架构总览

```
人类指令 "把仓库 A1 的 Nivea 补到货架 B3 第二层"
        │
        ▼
┌─────────────────────────────────────────────┐
│  System 2 — VLM 规划器 + 完成判断 (qwen-vl)    │  慢, 事件触发
│  文本地图 + 指令  →  子任务序列 (JSON)         │
│  传感器触发时  →  看头部前后帧 → 判断 verdict  │
└─────────────────────────────────────────────┘
        │  下发子任务; 按 verdict 推进/重试/重规划/中止
        ▼
┌─────────────────────────────────────────────┐
│  System 1 — 低层执行器                         │  快, 10Hz
│   navigate_to → 点目标导航 (分段直线)          │
│   turn_to     → 到达后转向, 面朝货架           │
│   pick_* / restock_* → VLA (Fetch 15-DoF)     │
└─────────────────────────────────────────────┘
```

- **导航分段**: 每个 `navigate_to` 是相邻 waypoint 间的一条直线; 长距离 = 多段链式导航。
  段内避障交给局部规划器 (iPlanner / NavDP)。
- **到达后转向**: 导航只保证到点, 不保证朝向; 到货架接近点后用 `turn_to` 面朝货架再操作。
- **闭环完成判断**: 便宜的传感器信号当**触发器**; VLA 子任务触发后调一次 VLM 看图判断
  (`IN_PROGRESS`/`SUCCESS`/`RETRY`/`REPLAN`/`ABORT`), 由此决定继续/前进/重试/重规划/中止。
  对标 G0 的 System-2 (2Hz 固定重规划), 但改成**事件触发**以省 API 调用。
  `--no-vlm-judge` 可退回纯传感器开环路径。

---

## 文件清单

| 文件 | 作用 |
|------|------|
| `configs/data/fetch/pd_joint_pos.yaml` | Fetch 数据配置 (15-DoF action/state + 3 路相机的 shape_meta 与 processor) |
| `configs/task/robobenchmart/fetch_lora_finetune.yaml` | 跨本体 LoRA 微调任务配置 |
| `scripts/convert_robobenchmart_to_lerobot.py` | RoboBenchMart h5 → LeRobot 数据转换 |
| `scripts/run/finetune_lora_fetch.sh` | 微调启动脚本 |
| `configs/maps/store_layout1.json` | 商超文本地图 (ASCII 网格 + 图例 + waypoints) |
| `scripts/deploy_supermarket.py` | 分层部署入口 (VLM 规划 + 顺序执行) |

---

## 步骤 1 — 生成 Fetch 演示数据 (RoboBenchMart)

在 RoboBenchMart 仓库里跑运动规划, 生成轨迹, 再 replay 出图像观测。

```bash
cd /home/lh/VLA/RoboBenchMart-main

# 1a. 生成场景 (按 RoboBenchMart 文档)
python scripts/generate_scene_continuous.py ds_continuous=small_scene

# 1b. 运动规划生成轨迹 (Fetch, 只存成功的)
python scripts/run_mp.py \
    -e PickToBasketContNiveaEnv \
    --scene-dir generated_envs/layout1/pick_nivea \
    -r ds_fetch_basket \
    -n 20 \
    --only-count-success

# 1c. replay 出 RGB 观测 (rgbd 模式同时存 RGB+depth, 我们只用 RGB)
python scripts/replay_trajectory.py \
    --traj-path generated_envs/layout1/pick_nivea/demos/motionplanning/*.h5 \
    --obs-mode rgbd \
    --save-traj
```

> Fetch (`ds_fetch_basket`) 有 2 路相机: `head_camera` (256×256) 和 `fetch_hand` (128×128)。
> 动作空间 15-DoF: base(3) + torso_lift(1) + arm(7) + head(2) + gripper(2)。

---

## 步骤 2 — 转换为 LeRobot 格式

把 replay 出来的 h5 转成 GalaxeaVLA 训练用的 LeRobot 数据集。

```bash
cd /home/lh/VLA/GalaxeaVLA-main

python scripts/convert_robobenchmart_to_lerobot.py \
    --h5-dir /home/lh/VLA/RoboBenchMart-main/generated_envs/layout1/pick_nivea/demos/motionplanning/ \
    --output-dir datasets/supermarket_fetch/pick_nivea \
    --task "pick Nivea and put to the basket"
```

相机映射 (Fetch 2 路 → G0 3 路):

```
head_camera (256×256) → head_rgb         → resize 224×224
fetch_hand  (128×128) → left_wrist_rgb   → resize 224×224
fetch_hand  (128×128) → right_wrist_rgb  → 复用同一路 (Fetch 只有 1 个腕部相机)
```

---

## 步骤 3 — LoRA 跨本体微调

```bash
# 单卡
bash scripts/run/finetune_lora_fetch.sh 1 \
    model.pretrained_ckpt=/path/to/G0Plus_260202 \
    data.dataset.dataset_dirs='["datasets/supermarket_fetch/pick_nivea"]'

# 4 卡, 调大 batch
bash scripts/run/finetune_lora_fetch.sh 4 \
    model.pretrained_ckpt=/path/to/G0Plus_260202 \
    data.dataset.dataset_dirs='["datasets/supermarket_fetch/pick_nivea"]' \
    model.batch_size=8
```

### 跨本体的关键: 维度变化如何处理

R1Lite (26-DoF action, 27-DoF state) → Fetch (15-DoF, 15-DoF)。只有输入/输出边界层的形状变了,
中间的语义层全部复用:

```
模块                        处理方式
─────────────────────────  ──────────────────────────────
ActionEncoder.linear_1      ❌ 形状变 (26→15) → 全量训练 (modules_to_save)
ActionDecoder 最后一层       ❌ 形状变 (26→15) → 全量训练
ProprioEncoder              ❌ 形状变 (27→15) → 全量训练
ActionEncoder.linear_2/3    ✅ 形状不变 → LoRA
Action Mixture (18 层)       ✅ 形状不变 → LoRA
Vision Tower + VLM (Gemma)  ✅ 不变 → LoRA
其余 ~97% 参数               ❄️ 冻结
```

配置里只需改两行 (已在 `fetch_lora_finetune.yaml` 写好):

```yaml
model_arch:
  action_dim:  ${data.processor.action_output_dim}   # 15
  proprio_dim: ${data.processor.proprio_output_dim}   # 15
```

---

## 步骤 4 — 分层部署

VLM 把指令拆成子任务序列, 顺序执行。地图是文本 (`configs/maps/store_layout1.json`)。

```bash
export DASHSCOPE_KEY=sk-xxxx     # 或用 --vlm-api-key 传入

python scripts/deploy_supermarket.py \
    --command "restock Nivea from warehouse A1 to shelf B3 layer 2" \
    --map-file configs/maps/store_layout1.json \
    --vlm-provider qwen --vlm-api-key $DASHSCOPE_KEY \
    --vla-ckpt /path/to/fetch_lora_ckpt \
    --mode simulate
```

参数:

| 参数 | 说明 |
|------|------|
| `--command` | 总指令 (自然语言) |
| `--map-file` | 文本地图 JSON |
| `--vlm-provider` | `qwen` / `gemini` / `openai` |
| `--vlm-api-key` | API key (也可读环境变量 `VLM_API_KEY`) |
| `--vla-ckpt` | Fetch LoRA 权重路径 (留空 = mock 动作, 用于测架构) |
| `--mode` | `simulate` (内置 SimEnv) / `real` (需接 ROS2 桥接) |

### 子任务类型

| type | 执行器 | 传感器触发器 | 触发后 |
|------|--------|-------------|--------|
| `navigate_to` | 点目标导航 (一段直线) | 到达 (距目标 <0.3m) | 直接完成 |
| `turn_to` | yaw PD 控制 | yaw 误差 <5° | 直接完成 |
| `pick_to_basket` | VLA | 夹爪闭合 | VLM 看图判断 |
| `pick_from_floor` | VLA | 夹爪闭合 | VLM 看图判断 |
| `restock_basket_to_shelf` | VLA | 夹爪张开 | VLM 看图判断 |

nav/turn 的到位/朝向无歧义, 触发即完成; VLA 子任务的传感器触发只表示"可能到终点",
要由 VLM judge 拍板 (见下方"闭环完成判断")。`--no-vlm-judge` 时所有子任务都退回触发即完成。

### 离线测架构 (无需 API / 权重)

`--vla-ckpt` 留空时 VLA 输出 mock 动作, `--mode simulate` 用内置 SimEnv。
配合一个返回固定计划的桩 planner, 可以验证完整调度链路 (规划→分段导航→到达转向→抓取→放置)。

<!-- PROMPT_SECTION -->

### VLM 提示词在哪

提示词不是单独文件, 是 `scripts/deploy_supermarket.py` 里的两个常量
(`# --- section: prompts ---` 段):

| 常量 | 作用 |
|------|------|
| `SYSTEM_PROMPT` | 5 种子任务定义 + 7 条规则 (导航分段 / 到达后转向) |
| `USER_TMPL` | 用户消息模板, 含 `{map_block}` 和 `{command}` 两个占位符 |

改提示词 = 改这两个常量。`StoreMap.render()` 负责把地图 JSON 填进 `{map_block}`。

### 输入给 VLM 的内容

```
SYSTEM: 角色 + 5 种 atomic 子任务 + 7 条规则
USER:   {map_block}  ← 文本地图 (ASCII 网格 + 图例 + waypoints + 当前位姿)
        {command}    ← 总指令
        (可选 1 帧当前头部 RGB, 让 VLM 看现场)
```

<!-- PROMPT_SECTION_2 -->

### 实际渲染出的 USER 消息 (示例)

机器人在 `(2.0, 3.0)` 朝东, 指令 "restock Nivea from warehouse A1 to shelf B3 layer 2":

```
=== STORE MAP (top-down, each cell = 1.0 m) ===
X -> east (right), Y -> north (up)
  y=6 [  ][  ][B1][B2][B3][  ]
  y=5 [  ][  ][  ][  ][  ][  ]
  y=4 [A1][A2][  ][C1][C2][F1]
  y=3 [A3][A4][R ][C3][C4][F2]
  y=2 [  ][  ][  ][  ][  ][  ]
  y=1 [  ][  ][  ][  ][  ][  ]
      x=0 x=1 x=2 x=3 x=4 x=5

=== LEGEND ===
  A1-A4: warehouse shelves (west). A1: Nivea, Fanta; ...
  B1-B3: retail shelves (north), 3 layers each. B3 layer 2 currently has empty slots
  ...

=== CORRIDOR WAYPOINTS (chain these into straight segments) ===
  warehouse_aisle: (2.0, 3.0)
  warehouse_mid: (2.0, 4.5)
  main_north_junction: (2.0, 5.0)
  ...

=== SHELF APPROACH POINTS (stand here to manipulate) ===
  warehouse_shelf_A1_front: pos=(2.0, 4.0), face=180deg
  retail_shelf_B3_front: pos=(4.0, 5.5), face=90deg

=== ROBOT NOW: at (2.0, 3.0) facing 0deg ===

=== HUMAN COMMAND ===
restock Nivea from warehouse A1 to shelf B3 layer 2
```

<!-- PROMPT_SECTION_3 -->

### VLM 的输出 (JSON 数组)

VLM 只输出 JSON, 每个元素三个 key: `type` / `instruction` / `target`:

```json
[
  {"type": "navigate_to", "instruction": "drive to the warehouse aisle midpoint", "target": "2.0,4.5"},
  {"type": "navigate_to", "instruction": "drive to the front of warehouse shelf A1", "target": "2.0,4.0"},
  {"type": "turn_to", "instruction": "turn to face warehouse shelf A1", "target": "180"},
  {"type": "pick_to_basket", "instruction": "pick the Nivea from shelf A1 into the basket", "target": "Nivea"},
  {"type": "navigate_to", "instruction": "drive to the main north junction", "target": "2.0,5.0"},
  {"type": "navigate_to", "instruction": "drive to the retail aisle east point", "target": "4.0,5.5"},
  {"type": "turn_to", "instruction": "turn to face retail shelf B3", "target": "90"},
  {"type": "restock_basket_to_shelf", "instruction": "place the Nivea onto shelf B3 layer 2", "target": "B3_layer2"}
]
```

`_parse()` 校验后转成 Subtask 序列 (丢弃非法 type, 设 max_steps):

```
[1] nav  navigate_to               target=2.0,4.5    ┐
[2] nav  navigate_to               target=2.0,4.0    ├ 多段直线导航 → A1
[3] nav  turn_to                   target=180        ┘ 到达后转向, 面朝货架
[4] vla  pick_to_basket            target=Nivea        抓取 (夹爪闭合=done)
[5] nav  navigate_to               target=2.0,5.0    ┐
[6] nav  navigate_to               target=4.0,5.5    ├ 多段直线导航 → B3
[7] nav  turn_to                   target=90         ┘ 到达后转向
[8] vla  restock_basket_to_shelf   target=B3_layer2    放置 (夹爪张开=done)
```

各字段去向:

| 字段 | 用途 |
|------|------|
| `type` | 选执行器: `navigate_to`/`turn_to`→nav, `pick`/`restock`→vla |
| `instruction` | pick/restock 时原样作为 `task` 文本喂给 VLA (语言条件) |
| `target` | navigate_to=坐标"x,y"; turn_to=角度; pick/restock=物品名 |

### 坐标怎么来的 (关键设计)

> **VLM 不从图像像素测量坐标, 而是从地图预设清单里"选"坐标。**

| 方式 | 精度靠谁 | 可靠性 |
|------|---------|--------|
| 给图 + 单位长度, VLM 数像素算坐标 | VLM 空间估算 | ❌ 误差常 >0.5m, 会撞货架 |
| VLM 从 waypoint 清单选 (本方案) | 地图标定 (人/SLAM) | ✅ 厘米级 |

ASCII 网格让 VLM 理解**拓扑**(A1 在西、B3 在北、走哪条道不穿墙);
waypoint 清单提供**精确坐标**。VLM 做的是"Nivea 在 A1 → 选
`warehouse_shelf_A1_front` → 抄它的坐标 (2.0,4.0)" —— 语义匹配 + 路径排序,
而不是空间测量。精度押在地图上, 不押在模型最弱的像素感知上。

### 完整数据流

```
SYSTEM(规则) + USER(文本地图 + 位姿 + 指令)
   │
   ▼  VLM 语义匹配 + 路径排序 (坐标从清单抄, 不测量)
JSON 数组 [{type, instruction, target}, ...]
   │
   ▼  _parse() 校验 → Subtask 序列
顺序执行; VLA 子任务靠"传感器触发 + VLM 看图判断"推进 (见下方闭环章节)
```

---

## 闭环完成判断 (trigger + VLM judge 状态机)

### 为什么不让 VLA 自己判断完成

VLA 内部确实有 VLM backbone, 但它判断不了任务完成 —— 三个原因:

```
1. 没有输出口
   VLA (galaxea_zero.py: infer_action) 只输出动作块 (bsz, horizon, action_dim),
   没有 "termination" 通道。action_dim=15 全是关节/底盘/夹爪, 没地方"说"做完了。
2. 没被训过
   训练是 (观测, 指令, 动作) 三元组 + flow matching 模仿动作, 全程无"完成"标签。
3. 看不到判断所需的信息  ← 最关键
   VLA 输入是单帧 (obs_size=1, 无历史) + 手腕特写, 为"精细控制"设计;
   判断"整个子任务做完没"需要场景级视角 + 时序历史 —— 那是 System 2 的输入。
```

更深一层: 让生成动作的模型给自己打"完成分", 是 generator 给自己当 verifier ——
校准差、有利益冲突。**独立的判断者信号更干净**, 这正是 G0 双系统 generator/verifier
分离的核心。所以完成判断交给 System 2 (这里是独立的 VLM judge 调用), 不问 VLA。

### 状态机

便宜的传感器信号当**触发器** (每步免费); VLA 子任务触发后, 调一次 VLM 看头部
**前后两帧** 给出 verdict, 由 verdict 驱动调度器:

```
        ┌──────── EXECUTING (VLA 跑当前子任务, 指令固定) ◄──┐
        │                  │                                │
        │          传感器触发器响 (夹爪/到位)                 │
        │                  ▼                                │
        │            VLM judge (头部前后帧)                  │
        │     ┌────────────┼────────────┬───────────┐       │
        ▼     ▼            ▼            ▼           ▼       │
   IN_PROGRESS  RETRY    SUCCESS      REPLAN      ABORT     │
   什么都不改  重试当前  下一个子任务  重规划剩余   中止       │
   (+cooldown) (≤预算)               (≤预算)               │
        └──────┴────────────┴────────────┴──────────────────┘
```

| verdict | 含义 | 动作 |
|---------|------|------|
| `IN_PROGRESS` | 符合预期, 还没到终点 (最常见) | 不换指令, 刷新基准帧继续跑; cooldown 后才再问 |
| `SUCCESS` | 到达终止状态 (入篮/上架) | 前进到下一个子任务 |
| `RETRY` | 失败但世界没坏 (抓空了, 物品还在) | 重跑当前子任务 (≤ `max_retries`) |
| `REPLAN` | 失败且世界变了, 剩余计划失效 (掉地上) | 从当前状态重新规划尾部 (≤ `max_replans`) |
| `ABORT` | 不可恢复 (撞了/够不到/超时) | 中止, 整体判失败 |

要点:
- **nav/turn 不调 VLM**: 到位/朝向无歧义, 传感器触发即完成。只有 VLA 子任务才看图判断。
- **cooldown 防刷屏**: `IN_PROGRESS` 后静默 N 步再问, 否则退化成高频 API 轮询 (烧钱)。
- **对标 G0**: G0 的 System-2 以 2Hz 固定频率重规划 (本地模型, 调用免费); 这里改成
  **事件触发** (API 模型按需调用省钱), 本质相同 —— 看观测决定 继续/前进/重试/重规划。
- **降级保护**: judge 调用/解析失败一律返回 `IN_PROGRESS` (宁可继续也不误前进/误中止)。
- `--no-vlm-judge` 退回纯传感器开环 (触发即完成, 无恢复)。

实现位置 (`scripts/deploy_supermarket.py`):

| 部件 | 位置 |
|------|------|
| `JudgeVerdict` 枚举 | 文件顶部任务分类后 |
| `JUDGE_SYSTEM_PROMPT` / `VLMPlanner.judge()` | prompts / planner 段 |
| 触发器 `Scheduler._trigger()` + judge 调用 `_execute()` | scheduler 段 |
| verdict 状态机 `Scheduler.run()` | scheduler 段 |

---

## 已知局限 (future work)

已实现 (本轮闭环改造):

| 曾经的局限 | 现状 |
|-----------|------|
| 开环: 规划一次, 不重规划 | ✅ `REPLAN` verdict 从当前状态重规划剩余子任务 (≤ `max_replans`) |
| 无完成度确认: 只看传感器 done | ✅ VLA 子任务触发后 VLM 看头部前后帧判断 verdict |
| 无掉落恢复 | ✅ `REPLAN` 可在尾部插入 `pick_from_floor` (取决于 VLM 重规划结果) |
| 子任务失败 = 整体失败 | ✅ `RETRY` (世界没坏时重试) / `REPLAN` (世界变了时重来) 分级恢复 |

仍待改进:

| 当前 | 改进方向 |
|------|---------|
| `pick_*` 触发器用"夹爪闭合" (中间事件, 非释放边沿) | 改成"抓了又松开"的释放边沿; 现靠 VLM judge 兜底 (闭合时判 IN_PROGRESS) |
| VLM judge 是通用 API, 未领域微调 | 像 G0-VLM 那样用机器人数据 SFT (论文 Table 1: 通用 VLM 当规划器准确率仅 15~55%) |
| 重规划 = 重新全量规划尾部 | 增量修复 (只插入恢复步骤, 不整段重排) |
| 腕部相机两路复用同一张图 | Fetch 加装第二个腕部相机, 或改双臂本体 |
| ABORT 仅判失败 | 接人工介入 / 报警 |

主链路 "VLM 分层规划 + 闭环完成判断 + 跨本体 VLA 执行" 已做通; 上面是进一步打磨方向。
