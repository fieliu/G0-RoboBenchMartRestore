#!/bin/bash
# LoRA fine-tuning G0Plus for Fetch robot (RoboBenchMart)
# Cross-embodiment: R1Lite (26D action) → Fetch (15D action)
#
# Usage:
#   bash scripts/run/finetune_lora_fetch.sh <GPU_NUM> [HYDRA_OVERRIDES...]
#
# Examples:
#   # Single GPU, default settings
#   bash scripts/run/finetune_lora_fetch.sh 1
#
#   # 4 GPUs, override batch size and LR
#   bash scripts/run/finetune_lora_fetch.sh 4 model.batch_size=8 model.learning_rate=1e-4
#
#   # 8 GPUs, higher LoRA rank
#   bash scripts/run/finetune_lora_fetch.sh 8 lora.rank=32 lora.alpha=64

export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

GPU=$1
ARGS=${@:2}

torchrun --standalone --nnodes 1 --nproc-per-node $GPU \
    scripts/finetune_lora.py \
    task=robobenchmart/fetch_lora_finetune \
    $ARGS
