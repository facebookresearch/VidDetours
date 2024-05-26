#!/bin/bash


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Uncomment and set the following variables correspondingly to run this script:

PROMPT_VERSION=plain

source ~/.conda/envs/detours/bin/activate

~/.conda/envs/detours/bin/deepspeed detours/train/train_mem.py \
    --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --version llama_2 \
    --data_path ${DATA_PATH} \
    --data_path_val ${DATA_PATH_VAL} \
    --data_path_test ${DATA_PATH_TEST} \
    --video_feats_folder ${VIDEO_FEATS_FOLDER} \
    --use_feats True \
    --tune_mm_mlp_adapter True \
    --task_name only_localization \
    --tune_localization_head True \
    --label_names "videos2_localization_labels" \
    --mm_vision_select_layer -2 \
    --eval_only False \
    --eval_output_dir "checkpoints" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir ./checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --prediction_loss_only False \
    --eval_steps 0.05 \
    --save_strategy "steps" \
    --save_steps 0.05 \
    --save_total_limit 5 \
    --load_best_model_at_end=True \
    --metric_for_best_model="Mean IoU" \
    --greater_is_better=True \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "tensorboard"
