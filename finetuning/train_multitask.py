import os
import sys
import json
import warnings
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import torch
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    AutoConfig,
)
from transformers.utils import send_example_telemetry
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel, get_peft_model, TaskType, LoraConfig

from finetuning.task_config import task_config
from finetuning.dataset import SupervisedDataset
from finetuning.compute_metrics import compute_metrics


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "Choose from [eager, sdpa, flash_attention_2]"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    rank: int = field(
        default=64,
        metadata={"help": "Rank of LoRA adapters. LoftQ does not require this config. Used for fp16 LoRA or QLoRA."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )


@dataclass
class DataTrainingArguments:
    task: Optional[str] = field(default=None)
    data_dir: Optional[str] = field(default=None)
    train_dataset: Optional[str] = field(default=None)
    eval_dataset: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    test_split: Optional[str] = field(default="validation")
    train_on_inputs: bool = field(default=False)
    max_length: Optional[int] = field(default=512)
    use_normalized_template: bool = field(default=False)
    temperature: Optional[float] = field(default=None)
    top_p: Optional[float] = field(default=None)
    top_k: Optional[float] = field(default=None)
    greedy_decoding: bool = field(default=False)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    assert data_args.task in {"commonsense", "math"}
    assert data_args.task in task_config, f"Unrecognized task: {data_args.task}"

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_translation", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, float16 training: {training_args.fp16}, "
        + f"bfloat16 training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    train_dataset_str = data_args.train_dataset

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        model_max_length=data_args.max_length,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token is None: # For Llama-3
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
    else:
        tokenizer.pad_token = tokenizer.unk_token

    # Load dataset
    train_datasets = task_config[data_args.task]["train_datasets"] if data_args.train_dataset is None else [data_args.train_dataset]
    eval_datasets = task_config[data_args.task]["eval_datasets"] if data_args.eval_dataset is None else [data_args.eval_dataset]

    train_dataset = SupervisedDataset(
        data_args.task, 
        os.path.join(data_args.data_dir, train_datasets[0]) if data_args.data_dir is not None else train_datasets[0], 
        tokenizer, 
        data_split="train", 
        seed=training_args.seed, 
        max_n_example=data_args.max_train_samples,
    )
    trigger_tokens = train_dataset.trigger_tokens

    all_eval_datasets = {}
    for eval_dataset in eval_datasets:
        test_splits = data_args.test_split.split(";")
        all_eval_datasets[eval_dataset] = {}
        for split in test_splits:
            raw_eval = SupervisedDataset(
                data_args.task, 
                os.path.join(data_args.data_dir, eval_dataset), 
                tokenizer, 
                data_split=split, 
                seed=training_args.seed, 
                max_n_example=data_args.max_eval_samples,
            )
            all_eval_datasets[eval_dataset][split] = [raw_eval, raw_eval.raw_dataset]
    eval_datasets = all_eval_datasets

    # Load model
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, 
        attn_implementation=model_args.attn_implementation
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.gradient_checkpointing:
        logger.info("Use gradient checkpointing with LoRA.")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    # PEFT
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
        model = get_peft_model(model, lora_config)
    elif model_args.adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            model_args.adapter_name_or_path,
            is_trainable=True,
            token=model_args.token,
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            model_args.model_name_or_path,
            subfolder='apiq_init',
            is_trainable=True,
            token=model_args.token,
        )
    logger.info(model)
    logger.info(model.print_trainable_parameters())

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics = train_result.metrics
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        model.eval()
        assert next(model.parameters()).is_cuda

        eval_results = {}
        for dataset_name in eval_datasets:
            # split evalset into chunks
            for split, (eval_dataset, data_items) in eval_datasets[dataset_name].items():
                generations, stats = compute_metrics(
                    data_args.task, 
                    dataset_name, 
                    model, 
                    tokenizer, 
                    eval_dataset, 
                    data_items,
                    trigger_tokens, 
                    None, 
                    training_args.per_device_eval_batch_size, 
                    None,
                    split, 
                    data_args.greedy_decoding, 
                    data_args.temperature,
                    data_args.top_p, 
                    data_args.top_k
                )

                # log
                eval_results.update(stats)
                generations = stats if generations is None else generations
                result_json_file_name = f"{training_args.output_dir}/{dataset_name}_{split}_outputs.json"
                with open(result_json_file_name, 'w') as json_file:
                    json.dump(generations, json_file, indent=4)

        # log final eval stats
        eval_results["mean"] = np.mean(list(eval_results.values()))
        result_json_file_name = f"{training_args.output_dir}/eval_results.json"
        with open(result_json_file_name, 'w') as json_file:
            json.dump(eval_results, json_file, indent=4)

if __name__ == "__main__":
    main()