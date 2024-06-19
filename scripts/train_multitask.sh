#!/bin/bash

BIT=2
SIZE=7
GS=64


TASK=commonsense # or "math" for arithmetic reasoning
MODEL=./model_zoos/Llama-2-${SIZE}b-hf-w${BIT}g${GS}-fake-sym   # or real
DATA_DIR=./dataset
OUTPUT_DIR=./results/${TASK}/Llama-2-${SIZE}b-hf-w${BIT}g${GS}-fake-sym
TOOL=./train_multitask.py

mkdir -p $OUTPUT_DIR
torchrun --nproc_per_node=1 $TOOL \
    --do_train \
    --do_eval \
    --model_name_or_path $MODEL \
    --task $TASK \
    --data_dir $DATA_DIR \
    --test_split test \
    --use_normalized_template \
    --max_length 512 \
    --seed 42 \
    --learning_rate 3e-4 \
    --max_grad_norm 1 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --warmup_ratio 0.1 \
    --greedy_decoding \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --disable_tqdm true \
    --report_to "none" \
    --remove_unused_columns false \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir 2>&1 | tee $OUTPUT_DIR/out