#!/bin/bash


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Uncomment and set the following variables correspondingly to run this script:

PROMPT_VERSION=plain

source ~/.conda/envs/detours/bin/activate

~/.conda/envs/detours/bin/deepspeed detours/train/train_xformers.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --version llama_2 \
    --data_path ${DATA_PATH} \
    --data_path_val ${DATA_PATH_VAL} \
    --data_path_test ${DATA_PATH_TEST} \
    --video_feats_folder ${VIDEO_FEATS_FOLDER} \
    --use_feats True \
    --task_name only_retrieval \
    --tune_retrieval_head True \
    --label_names "videos2_retrieval_labels" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 False \
    --output_dir ./checkpoints \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --prediction_loss_only False \
    --eval_steps 10 \
    --eval_only False \
    --eval_output_dir "checkpoints" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
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
