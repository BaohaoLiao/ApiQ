#!/bin/bash

BIT=2
SIZE=7
GS=64

MODEL=./model_zoos/Llama-2-${SIZE}b-hf-w${BIT}g${GS}-fake-sym   # or real
OUTPUT_DIR=./results/gsm8k/Llama-2-${SIZE}b-hf-w${BIT}g${GS}-fake-sym
TRAIN_TOOL=./train_gsm8k.py
TEST_TOOL=./test_gsm8k.py

mkdir -p $OUTPUT_DIR
torchrun --nproc_per_node=1 $TRAIN_TOOL \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --learning_rate 0.0003 \
  --seed 42 \
  --num_train_epochs 6 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --disable_tqdm true \
  --do_train \
  --report_to none \
  --overwrite_output_dir 2>&1 | tee $OUTPUT_DIR/train_out

python -u $TEST_TOOL \
  --model_name_or_path $MODEL \
  --ckpt_dir $OUTPUT_DIR \
  --batch_size 16 2>&1 | tee $OUTPUT_DIR/eval_out