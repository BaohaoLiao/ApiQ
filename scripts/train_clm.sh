#!/bin/bash

BIT=2
SIZE=7
GS=64

MODEL=./model_zoos/Llama-2-${SIZE}b-hf-w${BIT}g${GS}-fake-sym   # or real
OUTPUT_DIR=./results/wikitext/Llama-2-${SIZE}b-hf-w${BIT}g${GS}-fake-sym
TOOL=./train_clm.py

mkdir -p $OUTPUT_DIR
torchrun --nproc_per_node=1 $TOOL \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --learning_rate 0.0003  \
    --seed 42 \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --save_total_limit 1 \
    --load_best_model_at_end true \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --do_train --do_eval \
    --report_to none \
    --block_size 1024 \
    --disable_tqdm true \
    --overwrite_output_dir 2>&1 | tee $OUTPUT_DIR/out