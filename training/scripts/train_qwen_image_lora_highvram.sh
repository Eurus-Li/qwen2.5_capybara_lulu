#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$REPO_ROOT"

DATASET_BASE_PATH=${DATASET_BASE_PATH:-"$REPO_ROOT/training/dataset/qwen_image_lora/train"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/training/outputs/qwen_image_lora_highvram"}
MODEL_ID=${MODEL_ID:-"Qwen/Qwen-Image"}
TRAIN_REPEAT=${TRAIN_REPEAT:-20}
EPOCHS=${EPOCHS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_RANK=${LORA_RANK:-64}
MAX_PIXELS=${MAX_PIXELS:-1048576}

mkdir -p "$OUTPUT_ROOT"

accelerate launch external/DiffSynth-Studio/examples/qwen_image/model_training/train.py \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$DATASET_BASE_PATH/metadata.csv" \
  --max_pixels "$MAX_PIXELS" \
  --dataset_repeat "$TRAIN_REPEAT" \
  --model_id_with_origin_paths "$MODEL_ID:transformer/diffusion_pytorch_model*.safetensors,$MODEL_ID:text_encoder/model*.safetensors,$MODEL_ID:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs "$EPOCHS" \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path "$OUTPUT_ROOT/model" \
  --lora_base_model dit \
  --lora_target_modules to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1 \
  --lora_rank "$LORA_RANK" \
  --use_gradient_checkpointing \
  --dataset_num_workers 4 \
  --find_unused_parameters \
  --task sft
