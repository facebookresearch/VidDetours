#!/bin/sh


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --nodes=8                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=256           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 72:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name
#SBATCH --error=logs/%x-%j.err           # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

source ~/.conda/envs/detours/bin/activate

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
 detours/train/train_mem.py --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --version llama_2 \
    --data_path $DATA_PATH \
    --data_path_val $DATA_PATH_VAL \
    --data_path_test $DATA_PATH_TEST \
    --video_feats_folder $VIDEO_FEATS_FOLDER \
    --use_feats True \
    --task_name only_retrieval \
    --tune_retrieval_head True \
    --retrieval_aggregator "transformer" \
    --label_names "videos2_retrieval_labels" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --prediction_loss_only False \
    --eval_steps 0.05 \
    --save_strategy "steps" \
    --save_steps 0.05 \
    --save_total_limit 1 \
    --load_best_model_at_end=True \
    --metric_for_best_model="pr_auc" \
    --greater_is_better=True \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "tensorboard" \
    --deepspeed scripts/zero2.json'
