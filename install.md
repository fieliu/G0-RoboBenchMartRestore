# GalaxeaVLA 服务器配置教程

> 版本/命令/环境变量均来自项目 `pyproject.toml` 与上游 `README.md`（已核实）。
> 系统/驱动层面的要求是基于这些依赖推断的通用做法。
> 每一步都附 **✅ 验证** 命令，确认安装正确再进入下一步。

## 一、系统与硬件要求

```
操作系统:  Linux (Ubuntu 20.04/22.04 推荐)。WSL2 可行, 但 GPU 驱动走 Windows 侧。
Python:    严格 3.10  (pyproject: >=3.10.16,<3.11) —— 不能用 3.11/3.12
CUDA:      12.8  (torch 装的是 cu128 wheel)
GPU 驱动:  支持 CUDA 12.8 → NVIDIA driver ≥ 535 建议
GPU 显存:
   推理:        > 8 GB   (RTX 3090/4090)
   全量微调:    > 70 GB  (A100 80G / H20 96G)
   LoRA 微调:   远低于全量, A100 40G 可行
系统包:    ffmpeg (视频编解码必需)
```

## 二、配置步骤（每步带验证）

### 步骤 0：检查 GPU 和驱动

```bash
nvidia-smi
```
**✅ 验证**：能看到 GPU 型号 + `CUDA Version: 12.x`（≥12.8 最佳）。看不到先解决驱动。

### 步骤 1：装系统依赖

```bash
sudo apt update && sudo apt install -y ffmpeg git
ffmpeg -version    # ✅ 验证: 有版本号输出
```

### 步骤 2：安装 uv（项目指定的包管理器，勿用 conda）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env    # 或重开终端
uv --version                   # ✅ 验证: 有版本号
```

> 国内网络可在终端开头加镜像：
> ```bash
> export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
> export UV_PYTHON_INSTALL_MIRROR=https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download
> ```

### 步骤 3：同步依赖 + 安装项目

```bash
cd /home/lh/VLA/GalaxeaVLA-main
uv sync --index-strategy unsafe-best-match
source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[dev]"
```
**✅ 验证**（最关键一步）：
```bash
python --version                                  # 应是 3.10.x
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.7.1+cu128 True
```
`torch.cuda.is_available()` 必须为 `True`。若为 `False`：驱动不支持 cu128，或 WSL 里 GPU 没透传（最常见卡点）。

### 步骤 4：验证核心库导入

```bash
python -c "import transformers, peft, accelerate, diffusers; \
print('transformers', transformers.__version__, '| peft', peft.__version__)"
# 期望: transformers 4.57.1 | peft 0.18.0

python -c "import galaxea_fm; print('galaxea_fm OK')"
```
**✅ 验证**：无 ImportError，版本号对得上，项目包本身能导入（确认 `-e .` 生效）。

### 步骤 5：验证数据/视频工具链（转换 h5 要用）

```bash
python -c "import h5py, av, cv2, numpy; \
print('h5py', h5py.__version__, '| numpy', numpy.__version__)"
# 期望: numpy 1.26.4 (必须 1.x, 不是 2.x)
```
**✅ 验证**：`av`(PyAV) 能导入——转换脚本编码 mp4 必须靠它。

### 步骤 6：设置环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com                       # HF 国内镜像
export HF_DATASETS_CACHE=/home/lh/VLA/hf_cache                 # HF 缓存(空目录)
export GALAXEA_FM_OUTPUT_DIR=/home/lh/VLA/outputs              # 权重/日志输出
export GALAXEA_FM_DATASET_STATS_CACHE_DIR=/home/lh/VLA/stats   # 归一化统计缓存
export SWANLAB_API_KEY=<你的key>                               # 训练日志(可选)
export VLM_API_KEY=sk-xxxx                                     # VLM 规划器 key(部署用)
mkdir -p $HF_DATASETS_CACHE $GALAXEA_FM_OUTPUT_DIR $GALAXEA_FM_DATASET_STATS_CACHE_DIR
echo $GALAXEA_FM_OUTPUT_DIR    # ✅ 验证: 非空且目录存在
```

### 步骤 6.1：安装 mani_skill
uv pip install mani_skill
> 建议写进 `~/.bashrc`，避免每次重开终端丢失。

### 步骤 6.5：配置 VLM 规划器（仅部署需要，训练不需要）

部署 (`deploy_supermarket.py`) 的高层规划器走云端 VLM API。**只需给 provider + key 就能跑**，
不指定 model 则用默认模型。代码读环境变量 `VLM_API_KEY`，或用 `--vlm-api-key` 传入。

| provider | 默认模型 | 装库 | key 来源 |
|----------|---------|------|---------|
| `qwen`(默认) | `qwen-vl-max` | `uv pip install dashscope` | 阿里云百炼 |
| `gemini` | `gemini-2.0-flash` | `uv pip install openai` | Google AI Studio |
| `openai` | `gpt-4o` | `uv pip install openai` | OpenAI |

```bash
uv pip install dashscope     # 用 qwen 必装; gemini/openai 则装 openai
python -c "import dashscope; print('dashscope OK')"   # ✅ 验证
```

> 选型两条硬标准：① 必须是 **VLM（能看图）**——judge 要看头部前后帧，纯文本模型用不了；
> ② 指令跟随强、JSON 输出稳。规划任务不难，默认模型都够用。
> 想换具体型号才加 `--vlm-model <name>`。不配 key 也能跑（mock 规划，仅测架构）。

### 步骤 7：下载预训练权重

```bash
huggingface-cli download OpenGalaxea/G0-VLA G0Plus_3B_base \
    --local-dir ./ckpts/G0Plus_3B_base
ls -lh ./ckpts/G0Plus_3B_base/   # ✅ 验证: 有 .pt/.safetensors, 数 GB
```

### 步骤 8：端到端冒烟测试

```bash
cd /home/lh/VLA/GalaxeaVLA-main
python scripts/deploy_supermarket.py \
    --command "test" --map-file configs/maps/store_layout1.json \
    --mode simulate --vlm-provider qwen 2>&1 | head -20
```
**✅ 验证**：能进入调度循环、不报 ImportError/语法错误（mock 模式无 API key 时在规划处停是预期的，说明依赖链路通了）。

## 三、一键验证脚本

存为 `check_env.sh`，一次性核对全部：

```bash
#!/bin/bash
echo "=== Python ===" && python --version
echo "=== Torch+CUDA ===" && python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
echo "=== Core libs ===" && python -c "import transformers,peft,accelerate,diffusers,galaxea_fm; print('imports OK')"
echo "=== Data libs ===" && python -c "import h5py,av,cv2,numpy; print('numpy',numpy.__version__)"
echo "=== ffmpeg ===" && ffmpeg -version | head -1
echo "=== Env vars ===" && echo "OUT=$GALAXEA_FM_OUTPUT_DIR"
echo "=== GPU ===" && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```
全部无报错、`cuda: True`、版本号对得上 = 配置成功。

bash scripts/run/finetune_lora_fetch.sh 2 \
  model.batch_size=2 model.grad_accumulation_steps=4 \
  model.model_arch.pretrained_model_path=./ckpts/paligemma-3b-pt-224 \
  model.pretrained_ckpt=./ckpts/G0Plus_3B_base \
  logger.type=swanlab logger.mode=disabled \
  checkpointing_steps=500

# 检查是否使用了 CUDA 视频解码
cd ~/VLA/G0-RoboBenchMartRestore
grep -rnE "VideoDecoder|device=|torchcodec|decode" src/galaxea_fm/data/ .venv/lib/python3.10/site-packages/lerobot/ 2>/dev/null | grep -iE "device|VideoDecoder|cuda|cpu" | head


(robort_mart) root@lh:/home/lh/VLA/GalaxeaVLA-main# $P scripts/test_vlm_planning.py \
  --scene-dir /home/lh/VLA/RoboBenchMart-main/demo_envs/pick_to_basket \
  --env-name PickToBasketContNiveaEnv --out-dir vlm_plan_test \
  --command "把 Fanta 放进篮子" --target-product Fanta \
  --call-vlm --vlm-provider anthropic \
  --vlm-api-key "sk-eZj3ivmJ40XCbNrYJdDgb9mtRwcmlJdN6YFoiBS97hTpOlD0" \
  --vlm-base-url "https://www.packyapi.com" \
  --vlm-model "claude-opus-4-8"


python scripts/nav_sim_integration.py \
    --scene-dir /root/autodl-tmp/RoboBenchMartRestore/demo_envs/pick_to_basket \
    --env-name RestockBasketToShelfContDuffEnv \
    --command "把仓库里的Duff补货到商业区货架" \
    --planner grid \
    --device cuda:0 \
    --vlm-provider anthropic \
    --vlm-api-key "sk-eZj3ivmJ40XCbNrYJdDgb9mtRwcmlJdN6YFoiBS97hTpOlD0" \
    --vlm-base-url "https://www.packyapi.com" \
    --vlm-model "claude-opus-4-8"

export ANTHROPIC_AUTH_TOKEN="sk-eZj3ivmJ40XCbNrYJdDgb9mtRwcmlJdN6YFoiBS97hTpOlD0"
export ANTHROPIC_BASE_URL="https://www.packyapi.com"
export ANTHROPIC_MODEL="claude-opus-4-8"
python scripts/nav_sim_integration.py \
    --scene-dir /root/autodl-tmp/RoboBenchMartRestore/demo_envs/pick_to_basket \
    --env-name RestockBasketToShelfContDuffEnv \
    --command "把仓库里的Duff补货到商业区货架" \
    --planner grid \
    --device cuda:0

  cd /home/lh/VLA/GalaxeaVLA-main
  P=/home/lh/software/miniconda3/envs/robort_mart/bin/python
  export ANTHROPIC_BASE_URL=http://127.0.0.1:15721
  export ANTHROPIC_AUTH_TOKEN=sk-eZj3ivmJ40XCbNrYJdDgb9mtRwcmlJdN6YFoiBS97hTpOlD0
  export ANTHROPIC_MODEL=claude-opus-4-8

 

  $P scripts/nav_sim_integration.py \
    --scene-dir generated_envs/restock_scene \
    --env-name RestockFlowContDuffEnv \
    --command "把仓库里的Duff补货到商业区货架" \
    --planner grid \
    --device cuda:0


# 评估模型
source .venv/bin/activate

export ANTHROPIC_AUTH_TOKEN="sk-eZj3ivmJ40XCbNrYJdDgb9mtRwcmlJdN6YFoiBS97hTpOlD0"
export ANTHROPIC_BASE_URL="https://www.packyapi.com"
export ANTHROPIC_MODEL="claude-opus-4-8"

source .venv/bin/activate
export RBM_ROOT=/root/autodl-tmp/RoboBenchMartRestore
export GALAXEA_FM_OUTPUT_DIR=/root/autodl-tmp/G0-RoboBenchMartRestore/outputs
export GALAXEA_FM_DATASET_STATS_CACHE_DIR=/root/autodl-tmp/G0-RoboBenchMartRestore/stats
export GALAXEA_FM_DATA_DIR=/root/autodl-tmp/G0-RoboBenchMartRestore   # 数据集根目录
mkdir -p $GALAXEA_FM_OUTPUT_DIR $GALAXEA_FM_DATASET_STATS_CACHE_DIR

CKPT=/root/autodl-tmp/G0-RoboBenchMartRestore/pretain/model.pt
# 任务①  拿商品放篮子
python scripts/eval_robobenchmart.py \
    task=robobenchmart/fetch_lora_finetune \
    +ckpt_path=$CKPT \
    +eval_scene_dir=$RBM_ROOT/demo_envs/pick_to_basket \
    +eval_env_name=PickToBasketContNiveaEnv \
    +eval_num_traj=10 +eval_save_video=true

# 任务②  从地面捡商品
python scripts/eval_robobenchmart.py \
    task=robobenchmart/fetch_lora_finetune \
    +ckpt_path=$CKPT \
    +eval_scene_dir=$RBM_ROOT/demo_envs/pick_from_floor \
    +eval_env_name=PickFromFloorContNiveaEnv \
    +eval_num_traj=10 +eval_save_video=true

# 任务③  从篮子放回货架
python scripts/eval_robobenchmart.py \
    task=robobenchmart/fetch_lora_finetune \
    +ckpt_path=$CKPT \
    +eval_scene_dir=$RBM_ROOT/demo_envs/pick_to_basket \
    +eval_env_name=RestockBasketToShelfContNiveaEnv \
    +eval_num_traj=10 +eval_save_video=true

## 五、RoboBenchMart 仿真依赖（跑 nav_sim_integration / eval / 场景生成 需要）

> `scripts/nav_sim_integration.py` 等会 `import dsynth.envs`，依赖 RoboBenchMart 仓库 (`dsynth` 包)
> 及其底层仿真库 ManiSkill3。下面这些**装进当前项目的 `.venv`**（不是单独环境）。

### 5.1 安装缺失依赖（按报错顺序逐个补，已整理成一条）

```bash
cd ~/VLA/G0-RoboBenchMartRestore && source .venv/bin/activate

# RoboBenchMart 的全部依赖（来自 RoboBenchMartRestore/requirements.txt）
# 用阿里云镜像加速；unsafe-best-match 避免多源版本解析失败
uv pip install --index-strategy unsafe-best-match \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    mani_skill "scene-synthesizer[recommend]" usd-core "pyglet<2" \
    hydra-core trimesh==4.5.3 typeguard==4.4.2 pandas websockets msgpack "huggingface_hub[cli]"

# VLM 规划器 SDK（按用的 provider 装；anthropic=claude, dashscope=qwen, openai=gpt/gemini）
uv pip install -i https://mirrors.aliyun.com/pypi/simple/ anthropic dashscope openai
```

**遇到过的报错链（都是依赖没装齐）：**
```
ModuleNotFoundError: No module named 'mani_skill'         → 装 mani_skill
ModuleNotFoundError: No module named 'scene_synthesizer'  → 装 scene-synthesizer[recommend]
ModuleNotFoundError: No module named 'anthropic'          → 装 anthropic（用 claude/anthropic provider 时）
```
> 注：装 scene-synthesizer 会把 mujoco 升到 3.9.0（dm-control 依赖），一般无影响。

### 5.2 验证依赖链路通

```bash
# dsynth 在另一个仓库，需要把它的路径加进 PYTHONPATH
PYTHONPATH=/root/autodl-tmp/RoboBenchMartRestore \
    python -c "import dsynth.envs; print('dsynth.envs import OK')"
# ✅ 验证：输出 dsynth.envs import OK（环境注册成功）
```

### 5.3 运行仿真集成脚本

```bash
# 若报 No module named 'dsynth'，在命令前加 PYTHONPATH 指向 RoboBenchMart 仓库根目录
export PYTHONPATH=/root/autodl-tmp/RoboBenchMartRestore:$PYTHONPATH

python scripts/nav_sim_integration.py \
    --scene-dir generated_envs/restock_scene \
    --env-name RestockFlowContDuffEnv \
    --command "把仓库里的Duff补货到商业区货架" \
    --planner grid \
    --device cuda:0
```

> ⚠️ ManiSkill 仿真渲染需要 **Vulkan API**。若报 Vulkan/渲染相关错误，参考
> [ManiSkill 安装故障排查](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting)。

### 5.4 终端中文乱码（粘贴中文指令显示乱码时）

现象：`locale: Cannot set LC_CTYPE ...`，且 `locale -a` 里没有 `en_US.UTF-8`，只有 `C.utf8`。

```bash
# 方案 A（最快，系统已有 C.utf8）：
export LC_ALL=C.utf8
export LANG=C.utf8
# 写进 ~/.zshrc 永久生效

# 方案 B（生成 en_US.UTF-8，治本）：
apt-get update && apt-get install -y locales
locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
```
