#!/bin/bash

# vicuna-7b-v1.5
MODEL_PATH=$1
# llava-v1.5-7b MLP
LLAVA_PATH=$2

OUT_NAME=$3

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_PATH \
    --version v1 \
    --data_path /lisiyuan/jx/LLaVA/playground/data/llava_v1_5_mix665k.json \
    --image_folder /lisiyuan/jwy/LLaVA_v_ft/playground/data \
    --vision_tower /lisiyuan/jx/.cache/clip_vitl_336 \
    --pretrain_mm_mlp_adapter $LLAVA_PATH/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/$OUT_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --use_augment True \
    --augment_type "mixup" \
    --use_ranking True \
    --use_tome True \
    --tome_ratio 0 \
    --tome_merge_num 288 \
    --report_to wandb
