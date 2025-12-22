#!/bin/bash

# BERT-IMDB情感分类训练脚本

python run.py \
    --model_name_or_path ./bert-base-uncased \
    --data_path ./dataset/imdb/plain_text \
    --output_dir ./outputs/bert_imdb_model \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --eval_strategy steps \
    --save_strategy steps \
    --metric_for_best_model accuracy \
    --report_to tensorboard \
    --logging_dir ./logs \
    --seed 42 \
    --dataloader_num_workers 4 \
    --max_length 512 \
    --bf16