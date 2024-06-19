<h1 align="center">
    <img src="./imgs/logo.webp" width="114" height="114">
    <p>ApiQ</p>
</h1>
<h3 align="center">
    <p>Finetuning of 2-Bit Quantized Large Language Model</p>
</h3>
<h5 align="center">

[![arXiv](https://img.shields.io/badge/ApiQ-2308.13137-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2402.05147)
 <br>

</h5>

ApiQ is a framework for quantizing and finetuning an LLM in low-bit format. It can:

- act as a post-trianing quantization framework, achieveing superior performance for various bit levels
- finetune the quantized model for saving GPU memory and obtaining superior finetuning results

<p float="left" align="middle">
  <img src="./imgs/overall.png">
</p>


## Supports
- ApiQ-bw for quantizing Llama-2 and Mistral-7B-v0.1 in 4, 3 and 2 bits
- Fintuning of real/fake quantized LLM on WikiText-2, GSM8K, 4 arithmetic reasoning tasks and eight commonsense reasoning tasks

## News

## Contents
- [Install](#install)
- [Model Zoo](#model-zoo)
- [Quantizaion](#quantization)
- [Finetuning](#finetuning)
- [Citation](#citation)

## Install
```
conda create -n apiq python=3.10 -y
conda activate apiq
git clone https://github.com/BaohaoLiao/ApiQ.git
cd ApiQ
pip install --upgrade pip 
pip install -e .
```

If you want to finetune a real quantized LLM, we leverage the kernel from [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ#quick-installation). You can install AutoGPTQ and optimum as follows:
```
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install gekko
pip install -vvv --no-build-isolation -e .
pip install optimum>=0.20.0
```

## Model Zoo
We provide fake/real and symmetrically/asymmetrically quantized models at Huggingface.
- fake: The LLM's weights are still in FP16
- real: The LLM's weights are in GPTQ format
- symmetric: The quantization is symmetric, friendly to [vllm](https://github.com/vllm-project/vllm)
- asymmetric: The quantization is asymmetric

**Note**: 
- For the finetuning of real quantized LLM, you need to use the real and symmetric version, because there is a bug in AutoGPTQ for the asymmetric quantizaion (see [discussion](https://github.com/OpenGVLab/OmniQuant/issues/57)). 
- Fortunately, the difference between the symmetric and asymmetric quantization is very tiny. All results in the paper are from the asymmetric quantization.

## Quantization
1. Quantize an LLM with GPU as ./scripts/quantize.sh.
```
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
```
It will output some files in ```--save_dir```:
- ```peft.pth```: contain the PEFT parameters 
- ```lwc.pth```: contain the quantization parameters
- folder ```apiq_init```: contain necessary files for finetuning a PEFT model
- Other: The quantized version of LLM in FP16 format. tokenizer files, etc.

2. Evaluate a quantized LLM with ```peft.pth``` and ```lwc.pth```. After quantization, you can evaluate the model again with ```--resume```.
```
SIZE=7
BIT=2
GS=64

SAVE_DIR=./model_zoos/Llama-2-${SIZE}b-hf-w${BIT}g${GS}-fake-sym

python ./apiq/main.py \
    --model_name_or_path meta-llama/Llama-2-${SIZE}b-hf \
    --lwc --wbits ${BIT} --group_size ${GS} \
    --epochs 0 --seqlen 2048 --nsamples 128 \  # set epochs to 0
    --symmetric \
    --eval_ppl \
    --save_dir $SAVE_DIR  \
    --resume $SAVE_DIR
```

3. Convert the fake quantized LLM to a real quantized LLM in GPTQ format (**only work for symmetric quantization**):
```
SIZE=7
BIT=2
GS=64

RESUME_DIR=SAVE_DIR=./model_zoos/Llama-2-${SIZE}b-hf-w${BIT}g${GS}-fake-sym
SAVE_DIR=./model_zoos/Llama-2-${SIZE}b-hf-w${BIT}g${GS}-real-sym
mkdir -p $SAVE_DIR

python ./apiq/main.py \
    --model_name_or_path meta-llama/Llama-2-${SIZE}b-hf \
    --lwc --wbits ${BIT} --group_size ${GS} \
    --epochs 0 --seqlen 2048 --nsamples 128 \  # set epochs to 0
    --symmetric \
    --eval_ppl \
    --save_dir $SAVE_DIR  \
    --resume $RESUME_DIR
```

## Fnetuning
1. WikiText-2
```
bash ./scripts/train_clm.sh
```
2. GSM8K
```
bash ./scripts/train_test_gsm8k.sh
```
3. Arithmetic / commonsense reasoning
```
# Download the training and test sets
bash ./scripts/download_datasets.sh

# Finetune
bash ./scripts/train_multitask.sh
```

## Aknowledgement
- Our quantization code is based on [OmniQuant](https://github.com/OpenGVLab/OmniQuant)
- Our finetuning code is based on [LoftQ](https://github.com/yxli2123/LoftQ), [pyreft](https://github.com/stanfordnlp/pyreft) and [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)

## Citation
If you find ApiQ or our code useful, please cite our paper:
```
@article{ApiQ,
  title={ApiQ: Finetuning of 2-Bit Quantized Large Language Model},
  author={Baohao Liao and Christof Monz},
  journal={arXiv preprint arXiv:2402.05147},
  year={2024}
}
```

