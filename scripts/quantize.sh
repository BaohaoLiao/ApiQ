#!/bin/bash

SIZE=7
BIT=2
GS=64

SAVE_DIR=./model_zoos/Llama-2-${SIZE}b-hf-w${BIT}g${GS}-fake-sym
mkdir -p $SAVE_DIR

python ./apiq/main.py \
    --model_name_or_path meta-llama/Llama-2-${SIZE}b-hf \
    --lwc --wbits ${BIT} --group_size ${GS} \
    --epochs 20 --seqlen 2048 --nsamples 128 \
    --peft_lr 0.0005 --peft_wd 0.1 --lwc_lr 0.005 --lwc_wd 0.1 \
    --symmetric \
    --eval_ppl \
    --aug_loss \
    --save_dir $SAVE_DIR