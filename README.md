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

If you want to finetune a real quantized LLM, we leverage the kernel from [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ#quick-installation). You should install AutoGPTQ and optimum as follows:
```
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install gekko
pip install -vvv --no-build-isolation -e .
pip install optimum>=0.20.0
```

## Model Zoo

## Quantization


## Fnetuning

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

