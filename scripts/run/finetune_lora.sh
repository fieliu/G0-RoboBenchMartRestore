#!/bin/bash
# LoRA fine-tuning script for GalaxeaVLA
# Usage:
#   bash scripts/run/finetune_lora.sh <GPU_NUM> <TASK_CONFIG> [HYDRA_OVERRIDES...]
#
# Example:
#   bash scripts/run/finetune_lora.sh 4 real/g0plus_r1lite_lora_finetune
#   bash scripts/run/finetune_lora.sh 8 real/g0plus_r1lite_lora_finetune lora.rank=32 lora.alpha=64

export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

GPU=$1
config=$2
ARGS=${@:3}

config="${config#configs/}"
config="${config#task/}"
config="${config%.yaml}"

torchrun --standalone --nnodes 1 --nproc-per-node $GPU scripts/finetune_lora.py task=$config $ARGS
