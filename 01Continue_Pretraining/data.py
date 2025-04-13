import os
from glob import glob
from itertools import chain
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


class dataset:
    def __init__(self, model_args, data_args, tokenizer, training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.training_args = training_args

        if self.data_args.block_size is None:
            self.block_size = self.tokenizer.model_max_length
            if self.block_size > 2048:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 2048. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
        else:
            if self.data_args.block_size > self.tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({self.data_args.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            self.block_size = min(self.data_args.block_size, self.tokenizer.model_max_length)

    # Preprocessing the datasets.
    def tokenize_function(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.block_size,
        )
        # Copy the input_ids to the labels for language modeling. This is suitable for both
        # masked language modeling (like BERT) or causal language modeling (like GPT).
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()

        return tokenized_inputs

    def tokenize_wo_pad_function(self, examples):
        return self.tokenizer(examples["text"])

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_text_function(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    def load_data(self):
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                cache_dir=self.model_args.cache_dir,
                streaming=self.data_args.streaming,
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    self.data_args.dataset_name,
                    self.data_args.dataset_config_name,
                    split=f"train[:{self.data_args.validation_split_percentage}%]",
                    cache_dir=self.model_args.cache_dir,
                    streaming=self.data_args.streaming,
                )
                raw_datasets["train"] = load_dataset(
                    self.data_args.dataset_name,
                    self.data_args.dataset_config_name,
                    split=f"train[{self.data_args.validation_split_percentage}%:]",
                    cache_dir=self.model_args.cache_dir,
                    streaming=self.data_args.streaming,
                )
        else:
            data_files = {}
            dataset_args = {}
            if self.data_args.train_file_dir is not None and os.path.exists(
                self.data_args.train_file_dir
            ):
                train_data_files = (
                    glob(f"{self.data_args.train_file_dir}/**/*.txt", recursive=True)
                    + glob(f"{self.data_args.train_file_dir}/**/*.json", recursive=True)
                    + glob(f"{self.data_args.train_file_dir}/**/*.jsonl", recursive=True)
                )
                logger.info(f"train files: {train_data_files}")
                # Train data files must be same type, e.g. all txt or all jsonl
                types = [f.split(".")[-1] for f in train_data_files]
                if len(set(types)) > 1:
                    raise ValueError(
                        f"train files must be same type, e.g. all txt or all jsonl, but got {types}"
                    )
                data_files["train"] = train_data_files
            if self.data_args.validation_file_dir is not None and os.path.exists(
                self.data_args.validation_file_dir
            ):
                eval_data_files = (
                    glob(f"{self.data_args.validation_file_dir}/**/*.txt", recursive=True)
                    + glob(f"{self.data_args.validation_file_dir}/**/*.json", recursive=True)
                    + glob(f"{self.data_args.validation_file_dir}/**/*.jsonl", recursive=True)
                )
                logger.info(f"eval files: {eval_data_files}")
                data_files["validation"] = eval_data_files
                # Train data files must be same type, e.g. all txt or all jsonl
                types = [f.split(".")[-1] for f in eval_data_files]
                if len(set(types)) > 1:
                    raise ValueError(
                        f"train files must be same type, e.g. all txt or all jsonl, but got {types}"
                    )
            extension = "text" if data_files["train"][0].endswith("txt") else "json"
            if extension == "text":
                dataset_args["keep_linebreaks"] = self.data_args.keep_linebreaks
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=self.model_args.cache_dir,
                **dataset_args,
            )

            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{self.data_args.validation_split_percentage}%]",
                    cache_dir=self.model_args.cache_dir,
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{self.data_args.validation_split_percentage}%:]",
                    cache_dir=self.model_args.cache_dir,
                    **dataset_args,
                )
        logger.info(f"Raw datasets: {raw_datasets}")

        # Preprocessing the datasets.
        if self.training_args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["validation"].features)

        with self.training_args.main_process_first(desc="Dataset tokenization and grouping"):
            if not self.data_args.streaming:
                if self.training_args.group_by_length:
                    tokenized_datasets = raw_datasets.map(
                        self.tokenize_wo_pad_function,
                        batched=True,
                        num_proc=self.data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not self.data_args.overwrite_cache,
                        desc="Running tokenizer on dataset",
                    )
                    lm_datasets = tokenized_datasets.map(
                        self.group_text_function,
                        batched=True,
                        num_proc=self.data_args.preprocessing_num_workers,
                        load_from_cache_file=not self.data_args.overwrite_cache,
                        desc=f"Grouping texts in chunks of {self.block_size}",
                    )
                else:
                    lm_datasets = raw_datasets.map(
                        self.tokenize_function,
                        batched=True,
                        num_proc=self.data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not self.data_args.overwrite_cache,
                        desc="Running tokenizer on dataset",
                    )
            else:
                if self.training_args.group_by_length:
                    tokenized_datasets = raw_datasets.map(
                        self.tokenize_wo_pad_function,
                        batched=True,
                        remove_columns=column_names,
                    )
                    lm_datasets = tokenized_datasets.map(
                        self.group_text_function,
                        batched=True,
                    )
                else:
                    lm_datasets = raw_datasets.map(
                        self.tokenize_function,
                        batched=True,
                        remove_columns=column_names,
                    )
            return lm_datasets
 
class fault_tolerance_data_collator:
    def __init__(self):
        pass

    def __call__(self, features: List) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        try:
            for k, v in first.items():
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in features])
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                    else:
                        batch[k] = torch.tensor([f[k] for f in features])
        except ValueError:  # quick fix by simply take the first example
            for k, v in first.items():
                if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([features[0][k]] * len(features))
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                    else:
                        batch[k] = torch.tensor([features[0][k]] * len(features))

        return batch


if __name__ == "__main__":
    from transformers import AutoTokenizer, HfArgumentParser, Seq2SeqTrainingArguments, Qwen2TokenizerFast

    from arguments import DataArguments, ModelArguments, ScriptArguments

    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()
    Tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    dataset = dataset(model_args=model_args, data_args=data_args, tokenizer=Tokenizer, training_args=training_args)
    lm_datasets = dataset.load_data()
    # lm_datasets.save_to_disk("./processed_data")

    # 数据规整器测试
    # batch = []
    # for i in range(10):
    #     batch.append(lm_datasets["train"][i])
    # fault_tolerance_data_collator()(batch)
