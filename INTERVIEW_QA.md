# 面试准备 · 双系统 VLA 仓储补货机器人项目

> 覆盖：项目本身 Q&A + 周边会被问到的其他方面（ML/DL 基础、VLA 领域、相关算法、行为题）。
> 标注约定：✅ 可自信讲　⚠️ 要诚实承认局限　🔥 高频追问/陷阱题

## 目录
- 一、项目整体与架构
- 二、G0 双系统原理
- 三、VLA 模型细节（System 1）
- 四、跨本体微调
- 五、分层部署与 VLM 规划
- 六、闭环完成判断（核心亮点）
- 七、数据与评估
- 八、局限与改进
- 九、其他方面：ML/DL 基础
- 十、其他方面：Transformer 与注意力
- 十一、其他方面：扩散与流匹配
- 十二、其他方面：多模态/VLM
- 十三、其他方面：模仿学习与 RL
- 十四、其他方面：VLA 领域横向对比
- 十五、其他方面：LoRA/PEFT 深挖
- 十六、其他方面：机器人学基础
- 十七、行为面试与项目软问题
- 十八、其他方面：经典 CV 模型与基础
- 十九、其他方面：经典 NLP / LLM 基础
- 二十、其他方面：优化与训练
- 二十一、其他方面：评估指标
- 二十二、其他方面：工程与系统

---

## 一、项目整体与架构

**Q1：一句话介绍项目。**
把预训练 G0Plus 双系统 VLA 通过 LoRA 跨本体微调迁移到 Fetch 机器人，并搭建 VLM 分层规划 + 传感器触发/VLM 裁决的闭环完成判断，实现仓储补货的长程移动操作。

**Q2🔥：为什么用分层架构，不用端到端长程 VLA？**
长程任务跨千步、分钟级，端到端难学难泛化。分层后复合任务的"组合"交给 VLM 零样本完成，只有原子操作技能才需要数据训练 VLA。加新复合任务只改提示词，不重训 VLA。

**Q3：System 1 / System 2 分别是什么？**
System 1 = VLA 执行器（快、高频、图像+指令+状态→动作块）；System 2 = VLM 规划器（慢、低频、指令→原子子任务）。

---

## 二、G0 双系统原理（arXiv 2509.00576）

**Q4：双系统怎么运行？频率？** Figure 1 频率金字塔：200Hz 底层控制 / 15Hz System1 VLA 出 action chunk / 2Hz System2 VLM 出子任务。两系统**异步**：System2 持续 2Hz 重规划"下一个子任务"，System1 在当前子任务下 15Hz 执行。

**Q5：两个系统同一个模型吗？怎么训练？** 不是，分开训练。System2 = Qwen2.5-VL 指令微调；System1 = PaliGemma backbone + flow matching action expert，三阶段训练。

**Q6：System2 输入输出？怎么判断完成？** 输入头部图 + 隔 1s 的 k 帧历史观测/动作 + 任务名（不喂手部相机/底层状态）；输出子任务语言。完成判断隐含在"持续决定下一个子任务"里——训练时给"子任务将结束/夹爪变化"的 key frame 更高采样权重，学会识别转换点。

**Q7：System2 训练数据怎么来？** §4.5 用 DeepSeek-R1 反向合成：喂任务名+历史/当前/下一子任务，让 R1 推理出"人类会怎么说"+"机器人怎么回应"。

**Q8🔥⚠️：通用 VLM API 当 System2 vs G0 微调版差距？** Table 1：通用模型当规划器准确率仅 15~55%，微调后 G0-VLM 到 75~83%。我的场景固定（地图/物品给定），通用 VLM 靠常识够用，但理想也该领域微调——工程成本权衡。

---

## 三、VLA 模型细节（System 1）

**Q9：动作怎么生成？为什么 flow matching 不用回归？** VLM backbone 编码图像+指令+本体→KV cache；action expert 以 KV cache 为条件，从噪声 flow matching 积分出连续动作块。flow/扩散能建模多模态动作分布（同一状态多种合理动作），回归会模式平均。

**Q10：VLA 有历史帧吗？** 没有时间历史（obs_size=1, past_action_size=0）。输入 3 路相机当前帧+当前状态+指令；时间连贯靠输出端 action chunk（一次预测 horizon 步未来动作）。`pixel_values` 的 T 维装的是 3 路相机不是 3 个时间步。

**Q11🔥：VLA 内部也有 VLM，为什么不让它自判完成？** 三点：①没有输出口（只输出动作块，无 termination 通道）；②没被训过（无完成标签）；③看不到判断所需信息（单帧+手腕特写，判断任务级完成需场景级+时序，那是 System2 的输入）。更深：让生成者给自己打完成分=generator 当自己的 verifier，校准差、有利益冲突。独立判断者信号更干净——generator/verifier 分离。

**Q12🔥：指令执行期间不变，VLA 怎么往下做？** 指令是目标不是单步命令。每次推理图像/状态在变（伸手→抓→抬→移→放），指令固定动作照样推进。VLA 学的是"给定指令整条轨迹长什么样"。

---

## 四、跨本体微调

**Q13：26-DoF→15-DoF 维度变了怎么处理？** 只边界层形状变，中间语义层全复用：ActionEncoder.linear_1 / ActionDecoder 末层 / ProprioEncoder → 全量训练；Action Mixture / Vision Tower / Gemma → LoRA；其余 ~97% 冻结。

**Q14：为什么 LoRA 不全量微调？** 小数据全量易破坏预训练知识、过拟合；LoRA 参数高效、显存友好（全量 >70G，LoRA 在 40G 可行）。rank16/alpha32，lr 5e-4。

**Q15：归一化统计量？** use_pretrained_norm_stats=False，从 Fetch 数据重算——动作分布不同必须重估。

**Q16：coarse_task / task？** G0 训练吃双指令 `[High]:{coarse_task},[Low]:{task}`。部署时总指令作 coarse_task、当前子任务作 task 透传，对齐训练分布。

---

## 五、分层部署与 VLM 规划

**Q17：VLM 怎么把指令变子任务？** 文本地图（ASCII 网格+图例+waypoint 清单）+指令 → JSON 子任务序列（type/instruction/target）→ `_parse` 校验转 Subtask。

**Q18🔥：坐标怎么来的？让 VLM 看图算吗？** 不让 VLM 从像素估坐标（最弱、误差 >0.5m 会撞架）。改成从地图 waypoint 清单"选"坐标：网格给拓扑，清单给厘米级坐标。VLM 做语义匹配+路径排序，精度押地图标定不押模型像素感知。

**Q19：导航为什么不用 VLA？** 导航是局部规划器（NavDP/iPlanner）几何控制，无学习成分、不训练。navigate_to 分段直线 + turn_to 到达后转向。

**Q20：receding horizon？为什么每 5 步重推理？** VLA 出 16/32 步 chunk 只执行前 5 步就重推理（replan_steps=5），用最新观测纠偏避免开环累积误差。导航/转向每步闭环（n_exec=1）。

---

## 六、闭环完成判断（核心亮点）

**Q21：怎么判断完成？sim vs 真机？** 仿真有 ground-truth done；真机没有，必须自己估——这是 sim2real 核心 gap。真机用传感器触发器+VLM 裁决。

**Q22🔥：详述触发器+judge 状态机。** 传感器触发（夹爪/到位/yaw）→ VLM 看头部前后帧 → 五态：IN_PROGRESS（不改+刷新基准帧继续+cooldown）/SUCCESS（前进下一子任务）/RETRY（世界没坏重试）/REPLAN（计划失效重规划）/ABORT（不可恢复中止）。

**Q23🔥：何时重规划何时只重试？** 决定的不是"成没成功"，是"世界还符不符合剩余计划假设"。抓空但物品还在→RETRY（重规划是浪费）；物品掉地上→REPLAN（前提被破坏，需插 pick_from_floor）。

**Q24：为什么事件触发不用 G0 的 2Hz？** G0 本地模型 2Hz 免费；我用 API 固定轮询慢且烧钱、中途 99% 返回"还在做"是浪费。事件触发省调用低延迟，本质相同。

**Q25：cooldown？** 判 IN_PROGRESS 后静默 N 步再问，否则退化成高频 API 轮询。

**Q26：judge 喂什么图？** 头部相机动作前后两帧对比（场景级，能看物品到位）。手部特写看不到全局不适合判进度；想确认抓取可加一帧手部看夹爪。

**Q27：judge 失败怎么办？** 降级返回 IN_PROGRESS——宁继续不误判。

---

## 七、数据与评估

**Q28：训练用多少数据？** 3 个 VLA 技能（导航不训）：pick_to_basket ~90 / restock ~90 / pick_from_floor ~60，共 ~240 条。锚点：论文 post-training 每任务 ≤100。

**Q29🔥：同量数据单物品堆量 vs 多物品少量？** 多物品少量。1物品×300背轨迹换物品崩；3物品×100同量但泛化。优先铺多样性。

**Q30：验证集？** val_set_proportion=0.05 自动按 episode 留。但 val loss 只当训练没崩哨兵，不选模型——模仿学习 loss 与成功率相关弱。

**Q31🔥：靠什么判好坏？为什么必须闭环评估？** 开环比动作 MSE 会累积误差，val loss 低≠抓得起。闭环=放进仿真真跑数成功率，唯一可信。闭环不预生成轨迹，只生成 test 场景现场 rollout，跑 20~30 episode，必须用 test 场景（未见布局）否则虚高。

**Q32🔥：闭环完成判断 vs 闭环评估一回事吗？** 不是。完成判断=部署时实时切子任务（运行时）；闭环评估=离线数成功率（开发时）。

**Q33：复合任务要训练数据吗？** 不要。VLM 零样本组合，RBM 里复合任务也只评估不生成训练数据。只训原子技能。

**Q34：240 条在 A100 40G 训多久？** ~12~20h（中位 16h）。决定因素是总帧数不是轨迹数。显存紧降 batch 升 grad_accum；想快先跑 5 epoch 看闭环。

---

## 八、局限与改进（主动暴露=加分）

**Q35⚠️：最大局限？** judge 是通用 API 未微调；pick 触发器用夹爪闭合是中间事件非释放边沿（靠 judge 兜底）；重规划是全量重排非增量；Fetch 单腕相机两路复用同图。

**Q36：更多资源下一步？** ①机器人数据 SFT 规划器；②触发器改释放边沿；③跑完整闭环评估报 train/test/OOD；④增量重规划。

---

## 九、其他方面：ML/DL 基础

**Q37：过拟合怎么判断、怎么缓解？** 训练 loss 降但验证 loss 升/成功率不涨。缓解：更多/更多样数据、正则（weight decay/dropout）、early stop、LoRA 限制可训参数、数据增强。本项目靠多物品多场景 + 冻结 97% 参数抗过拟合。

**Q38：batch size 与学习率的关系？** 大 batch 梯度估计更稳可配更大 lr（常用线性缩放）。本项目 batch4+grad_accum2=有效8；显存不够时降 batch 升 grad_accum 保持有效 batch 不变。

**Q39：梯度累积原理？** 多个 micro-batch 的梯度累加后再 step，等效大 batch 但峰值显存只占一个 micro-batch。代价是前后向次数不变、速度不提升。

**Q40：bf16/混合精度？** bf16 指数位同 fp32、动态范围大不易溢出，精度位少；省显存提速，主流大模型训练默认。注意 loss scaling（fp16 才需要，bf16 通常不需）。

**Q41：归一化（z-score）为什么重要？** 不同维度量纲差异大时，不归一化会让大量纲维度主导 loss。动作/状态 z-score 后各维同尺度，训练更稳。跨本体必须用目标本体数据重算。

---

## 十、其他方面：Transformer 与注意力

**Q42：自注意力复杂度？长序列怎么办？** O(n²) 序列长度。优化：FlashAttention（IO 感知、不省复杂度省显存带宽）、稀疏/线性注意力、滑窗。VLA 里图像 token 多，相机数×patch 数决定序列长度。

**Q43：KV cache 是什么？本项目哪里用到？** 自回归推理缓存历史 K/V 避免重复计算。G0 里 action expert 以 VLM 的 KV cache 为条件——VLM 算一次，action expert 多步去噪都复用，省算力。

**Q44：causal mask 在本项目的作用？** galaxea_zero 里 build_causal_mask 控制 vlm/proprio/action 三段 token 的可见关系，保证 action token 能看到 vlm+proprio 条件。

**Q45：位置编码？** 三段 token（vlm/proprio/action）各有 position_ids。RoPE/绝对编码各有取舍，VLA 里关键是区分模态段而非纯序列位置。

---

## 十一、其他方面：扩散与流匹配

**Q46🔥：flow matching 和 diffusion 的关系与区别？** 都从噪声生成目标。Diffusion 学逐步去噪的得分；flow matching 学一条从噪声到数据的概率流 ODE 的速度场，训练目标是回归速度向量。flow 路径更直、推理积分步数更少、更稳。

**Q47：为什么动作生成适合用 flow/diffusion 而非回归？** 动作分布多模态（同一状态多种合理动作），回归 MSE 会取平均落到不可行的中间值；生成式建模整个分布，能采样出某一个合理模态。

**Q48：psi_t / 插值在代码里？** galaxea_zero 里 x0=噪声、x1=动作、psi_t 是 t 时刻插值 `τ·x1+(1-τ)·x0`，模型预测从插值点指向数据的速度场，flow matching loss 监督它。

**Q49：推理时怎么从噪声得到动作？** 从随机噪声出发，沿学到的速度场积分若干步（欧拉等）到 t=1 得到动作块。步数是速度/质量权衡。

**Q50：action_expert_adaptive_mode / time_cond？** 时间步 t 通过 time_embedding 注入，告诉网络当前在去噪/积分的哪一步。adaptive 模式下用 AdaLN 类机制调制，否则拼接到输入。

---

## 十二、其他方面：多模态 / VLM

**Q51：SigLIP/CLIP 怎么把图像变 token？** 图像分 patch → 视觉编码器 → patch embedding 序列，经投影层对齐到语言 token 维度，与文本 token 一起进 Transformer。G0 用 SigLIP + 单层 MLP projector。

**Q52：PaliGemma 是什么？为什么 VLA 选它当 backbone？** SigLIP 视觉编码器 + Gemma 语言模型的开源 VLM。适合做 VLA backbone：体量适中、视觉-语言对齐好、易接 action expert。

**Q53：多模态怎么融合？early/late fusion？** VLA 是 early fusion——图像 token + 文本 token + 状态 token 拼成一个序列进同一个 Transformer 做注意力，而非各自编码后末端拼接。

**Q54：指令怎么影响动作？** 文本 token 与图像 token 在 backbone 里注意力交互，action expert 读这个融合表征（KV cache）条件生成。所以换指令=换条件=换动作分布。

---

## 十三、其他方面：模仿学习与 RL

**Q55🔥：这是模仿学习还是 RL？区别？** 模仿学习（behavior cloning）——从专家演示监督学动作，无奖励、无环境交互。RL 靠奖励试错。BC 简单稳定但有分布漂移问题（见下）。

**Q56🔥：BC 的分布漂移/复合误差怎么缓解？** 测试时策略自己的动作把状态带到训练没见过的分布，误差累积。缓解：receding horizon 用新观测纠偏、更多覆盖性数据、DAgger（交互式补数据）。本项目用 receding horizon + 闭环判断纠偏。

**Q57：为什么不用 RL？** 真机 RL 样本成本高、不安全；操作任务有大量演示数据，BC 性价比高。VLA 主流是大规模 BC 预训练。

**Q58：action chunking 为什么有用？** 一次预测多步未来动作，减少决策频率、缓解复合误差、动作更平滑连贯（ACT/pi0 的关键设计）。

---

## 十四、其他方面：VLA 领域横向对比

**Q59🔥：知道哪些 VLA 工作？G0 和它们什么关系？** RT-1/RT-2（Google，token 化动作）、OpenVLA（开源、自回归离散动作）、pi0（流匹配 action expert）、pi0.5、Octo（小型、扩散头）、Helix/GR00T（双系统）。G0 = pi0 式 flow matching 执行器 + 双系统规划，对标 Helix/GR00T 的快慢分层。

**Q60：自回归离散动作 vs 流匹配连续动作，优劣？** 离散（FAST tokenizer + 交叉熵）：复用 LM 训练范式、易和语言统一，但量化损失、长 chunk token 多。连续（flow matching）：动作精度高、多模态，但需额外 action expert。G0 Stage1 用离散预训练 VLM，Stage2 换流匹配 action expert——两者都用上了。

**Q61：双系统（Helix/G0）vs 单体 VLA，取舍？** 双系统：高层慢推理+低层快控制，长程任务强，但要协调两个模型。单体：简单统一，但长程规划弱。

**Q62：pi0 和 G0 的核心相似点？** 都是 VLM backbone + flow matching action expert + action chunking + 共享注意力/KV cache。G0 在其上加了双系统规划和大规模开放世界数据集。

---

## 十五、其他方面：LoRA / PEFT 深挖

**Q63🔥：LoRA 原理？为什么有效？** 冻结原权重 W，加低秩旁路 W+BA（B、A 低秩）。假设微调的权重更新本身是低秩的，所以少量参数够用。推理可合并回 W 无额外延迟。

**Q64：rank 和 alpha 怎么选？** rank 越大容量越强但越易过拟合/越费显存；alpha 是缩放（有效缩放 alpha/rank）。本项目 rank16/alpha32（缩放=2）是常见配置。

**Q65：哪些层加 LoRA？** 通常注意力的 q/k/v/o 投影；本项目还覆盖 Vision Tower 和 Action Mixture。维度变化的边界层不能用 LoRA（形状对不上），必须全量训练。

**Q66：LoRA vs 全量 vs adapter/prefix？** LoRA 无推理延迟、可合并；adapter 加层有延迟；prefix/prompt tuning 改输入不改权重、容量更小。LoRA 是当前主流。

**Q67：vlm_lr_multiplier 0.5 为什么？** 给 VLM 部分更小学习率，保护预训练视觉-语言知识，让动作相关层学得更激进些。

---

## 十六、其他方面：机器人学基础

**Q68：自由度（DoF）指什么？Fetch 15-DoF 怎么分？** 独立可控关节数。Fetch：base(3) + torso_lift(1) + arm(7) + head(2) + gripper(2)。R1Lite 26-DoF（双臂+躯干+底盘）。

**Q69：动作空间 pd_joint_pos 是什么？** 位置控制——动作=目标关节位置，底层 PD 控制器跟踪。区别于速度/力矩控制。Fetch 数据用绝对关节位置，所以无需相对变换。

**Q70：本体感知 proprioception 指什么？** 机器人自身状态（关节角、夹爪开合、底盘位姿），区别于外部感知（相机）。VLA 同时吃图像和 proprio。

**Q71：SE(2)/SE(3) 位姿？导航 waypoint 的 (x,y,yaw)？** SE(2)=平面位姿(x,y,yaw)，SE(3)=空间位姿(平移+旋转)。导航在 SE(2) 平面，waypoint 是 (x,y) + 朝向角。

**Q72：四元数 vs 欧拉角？** 欧拉角直观但有万向锁、插值不连续；四元数无奇异、插值平滑，机器人/图形学常用。turn_to 用 yaw 角（平面单自由度，欧拉够用）。

**Q73：什么是闭环控制 vs 开环？** 闭环用反馈持续纠偏（导航/转向每步读位姿）；开环按预定序列执行不看反馈。本项目 VLA receding horizon 是闭环，完成判断从开环升级到 VLM 闭环。

---

## 十七、行为面试与项目软问题

**Q74：这个项目你最大的收获/难点？** 收获：理解了双系统 VLA 的 generator/verifier 分离思想，以及 sim2real 在"完成判断"上的具体 gap。难点：搞清楚 G0 哪些是论文设计、哪些是我的简化（如事件触发 vs 2Hz），并诚实标注。

**Q75：哪个设计决策你最有信心？** waypoint 选取替代像素估坐标——把精度押在地图标定而非模型最弱的空间感知，是想清楚了模型能力边界后的决策。

**Q76：如果重做会改什么？** 先把闭环评估跑出真实成功率再写简历；规划器用领域数据微调而非通用 API；触发器一开始就用释放边沿而非夹爪闭合。

**Q77：怎么验证你的改动是对的？** 编译 + 端到端 sim 测试（如 coarse_task 透传、状态机五态转移都写了测试验证），而非只看代码。

**Q78：这个项目和岗位的关系？** 覆盖了具身智能/VLA 的完整链路：预训练模型迁移、多模态、生成式动作、分层决策、sim2real——能快速上手 VLA 相关研发。

**Q79🔥：诚实题——项目跑通了吗？成功率多少？** 如实说：架构链路端到端跑通（编译+sim 验证），跨本体微调配置就绪；闭环成功率还在评估中/待补。不编数字——编了追问必翻车。

---

## 附录：考前速查（最该练的 8 题）

| # | 题 | 一句话钥匙 |
|---|---|----------|
| Q11 | VLA 为何不能自判完成 | 无输出口/没训过/看不到时序，generator≠verifier |
| Q18 | 坐标怎么来 | 从地图清单选，不从像素估，精度押地图 |
| Q23 | 重试 vs 重规划 | 看世界是否破坏剩余计划，不是看成没成功 |
| Q31 | 为何必须闭环评估 | 开环误差累积，val loss≠成功率 |
| Q46 | flow vs diffusion | 学速度场 vs 学得分，flow 路径直步数少 |
| Q55 | IL vs RL + 分布漂移 | BC 学演示，receding horizon 纠偏 |
| Q59 | VLA 横向 | pi0/OpenVLA/Helix，G0=流匹配+双系统 |
| Q79 | 诚实题 | 链路通，成功率待补，不编数字 |

> 提示：被问到不确定的，先讲清"代码证实的"和"论文/推测的"边界，比硬答更显专业。

---

## 十八、其他方面：经典 CV 模型与基础

**Q80：CNN 为什么适合图像？卷积的归纳偏置？** 局部连接（局部感受野）+ 权值共享（平移不变）+ 层级特征。归纳偏置=局部性和平移等变，小数据比 ViT 更省样本。

**Q81：CNN vs ViT 区别与取舍？** CNN 有强归纳偏置、小数据好；ViT 全局注意力、大数据上限高但需更多数据/正则。VLA 的视觉编码器（SigLIP）是 ViT 系。

**Q82：ResNet 残差连接解决什么？** 梯度消失/退化问题，让深层网络可训；恒等映射使梯度直通。

**Q83：BatchNorm vs LayerNorm？为什么 Transformer 用 LN？** BN 跨 batch 统计，依赖 batch size、序列任务不稳；LN 在单样本特征维归一，与 batch 无关，适合变长序列/小 batch。Transformer 用 LN。

**Q84：目标检测两阶段 vs 一阶段？** 两阶段（Faster R-CNN）先选 proposal 再分类，准但慢；一阶段（YOLO/SSD）直接回归，快。本项目导航避障若用感知会涉及。

**Q85：图像分割语义/实例/全景区别？** 语义=每像素类别（不分个体）；实例=区分个体；全景=两者合并。

---

## 十九、其他方面：经典 NLP / LLM 基础

**Q86：word2vec / 词向量原理？** 用上下文预测词（CBOW）或词预测上下文（skip-gram），学出语义相近词向量相近。是早期分布式表示。

**Q87：BERT vs GPT？** BERT 双向编码器、masked LM、适合理解任务；GPT 单向解码器、自回归、适合生成。VLA 的语言 backbone（Gemma）是 GPT 式解码器。

**Q88：什么是 tokenizer？BPE？** 把文本切成子词单元。BPE 按频率合并字符对，平衡词表大小和 OOV。G0 Stage1 还用 FAST tokenizer 把连续动作离散化。

**Q89🔥：LLM 的 SFT / RLHF / DPO 区别？** SFT=监督微调学指令跟随；RLHF=用人类偏好训奖励模型再 RL（PPO）；DPO=跳过奖励模型直接用偏好数据优化，更简单稳定。G0-VLM 用的是 SFT。

**Q90：温度 temperature / top-p 采样？** 温度调分布尖锐度（低=确定，高=多样）；top-p 从累积概率 p 的核内采样。本项目 VLM 规划用低温（0.1）求确定性。

**Q91：幻觉 hallucination 怎么缓解？** RAG 检索增强、约束输出格式、降温、要求引用。本项目让 VLM 从 waypoint 清单"选"而非"编"坐标，就是约束输出抗幻觉。

---

## 二十、其他方面：优化与训练

**Q92：SGD / Adam / AdamW 区别？** SGD+动量简单泛化好；Adam 自适应学习率收敛快；AdamW 把权重衰减和梯度更新解耦，是 Transformer 主流。本项目用 AdamW。

**Q93🔥：学习率 warmup + cosine 为什么？** warmup 避免初期大梯度震荡（尤其大模型/大 batch）；cosine 衰减后期精调。本项目 warmup_steps=500 + cosine。

**Q94：梯度裁剪作用？** 防梯度爆炸，裁剪范数上限，训练更稳。

**Q95：weight decay / L2 正则？** 惩罚大权重防过拟合，提升泛化。本项目 1e-2。

**Q96：梯度消失/爆炸成因与对策？** 深层链式连乘。对策：残差、归一化、合理初始化、梯度裁剪、ReLU 系激活。

**Q97：常见激活函数？为什么用 GELU/SiLU？** ReLU 简单但死神经元；GELU/SiLU 平滑、Transformer 常用。

---

## 二十一、其他方面：评估指标

**Q98：精确率/召回率/F1？什么时候看哪个？** Precision=预测正里真正比例；Recall=真正里被找出比例；F1 调和平均。漏检代价高看 Recall，误报代价高看 Precision。

**Q99：mAP / IoU（检测分割）？** IoU=交并比衡量框/掩码重叠；mAP=各类 AP 平均，检测主指标。

**Q100🔥：本项目为什么用成功率不用 loss？（呼应 Q31）** 模仿学习 loss 衡量动作拟合，但误差累积、loss 低≠任务成。成功率是任务级闭环指标，唯一可信。

**Q101：ROC-AUC vs PR-AUC？** AUC 衡量排序能力；类别不平衡时 PR 曲线更敏感。

---

## 二十二、其他方面：工程与系统

**Q102：模型部署优化手段？** 量化（int8/fp16）、蒸馏、剪枝、算子融合、KV cache、batching。VLA 真机要满足实时性（15Hz）。

**Q103：训练慢/显存不够怎么排查优化？** profile 找瓶颈；混合精度、梯度累积、梯度检查点、ZeRO/FSDP 分片、LoRA 减可训参数。

**Q104：分布式训练 DP vs DDP vs FSDP？** DP 单进程多卡（慢）；DDP 多进程各持完整模型梯度同步；FSDP/ZeRO 分片参数/优化器状态省显存，训大模型用。

**Q105：数据加载是瓶颈怎么办？** 多 worker、预取、缓存解码、用高效格式（LeRobot/视频编码）。本项目数据存 mp4 + parquet。
