import os
from glob import glob

import debugpy
from datasets import load_dataset
from loguru import logger
from template import get_conv_template
from transformers.trainer_pt_utils import LabelSmoother

try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


class dataset:
    def __init__(
        self, data_args, model_args, script_args, training_args, tokenizer
    ) -> None:
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer
        self.script_args = script_args
        self.training_args = training_args
        # Get datasets
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            self.raw_datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                cache_dir=self.model_args.cache_dir,
            )
            if "validation" not in self.raw_datasets.keys():
                shuffled_train_dataset = self.raw_datasets["train"].shuffle(seed=42)
                # Split the shuffled train dataset into training and validation sets
                split = shuffled_train_dataset.train_test_split(
                    test_size=self.data_args.validation_split_percentage / 100, seed=42
                )
                # Assign the split datasets back to raw_datasets
                self.raw_datasets["train"] = split["train"]
                self.raw_datasets["validation"] = split["test"]
        else:
            # Loading a dataset from local files.
            data_files = {}
            if self.data_args.train_file_dir is not None and os.path.exists(
                self.data_args.train_file_dir
            ):
                train_data_files = glob(
                    f"{self.data_args.train_file_dir}/**/*.json", recursive=True
                ) + glob(f"{self.data_args.train_file_dir}/**/*.jsonl", recursive=True)
                logger.info(f"train files: {train_data_files}")
                data_files["train"] = train_data_files
            if self.data_args.validation_file_dir is not None and os.path.exists(
                self.data_args.validation_file_dir
            ):
                eval_data_files = glob(
                    f"{self.data_args.validation_file_dir}/**/*.json", recursive=True
                ) + glob(
                    f"{self.data_args.validation_file_dir}/**/*.jsonl", recursive=True
                )
                logger.info(f"eval files: {eval_data_files}")
                data_files["validation"] = eval_data_files
            self.raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
            )
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in self.raw_datasets.keys():
                shuffled_train_dataset = self.raw_datasets["train"].shuffle(seed=42)
                split = shuffled_train_dataset.train_test_split(
                    test_size=float(self.data_args.validation_split_percentage / 100),
                    seed=42,
                )
                self.raw_datasets["train"] = split["train"]
                self.raw_datasets["validation"] = split["test"]
        self.max_length = self.script_args.model_max_length
        self.IGNORE_INDEX = (
            LabelSmoother.ignore_index
            if data_args.ignore_pad_token_for_loss
            else tokenizer.pad_token_id
        )
        self.prompt_template = get_conv_template(self.script_args.template_name)
        logger.info(f"Raw datasets: {self.raw_datasets}")

    def preprocess_function(self, examples):
        """
        Preprocessing the datasets.
            part of code modified from https://github.com/lm-sys/FastChat
        """
        input_ids_list = []
        attention_mask_list = []
        targets_list = []
        roles = ["human", "gpt"]

        def get_dialog(examples):
            system_prompts = examples.get("system_prompt", "")
            for i, source in enumerate(examples["conversations"]):
                system_prompt = ""
                if len(source) < 2:
                    continue
                data_role = source[0].get("from", "")
                if data_role == "system":
                    # Skip the first one if it is from system
                    system_prompt = source[0]["value"]
                    source = source[1:]
                    data_role = source[0].get("from", "")
                if data_role not in roles or data_role != roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]
                if len(source) < 2:
                    continue
                messages = []
                for j, sentence in enumerate(source):
                    data_role = sentence.get("from", "")
                    if data_role not in roles:
                        logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                        break
                    if data_role == roles[j % 2]:
                        messages.append(sentence["value"])
                if len(messages) % 2 != 0:
                    continue
                # Convert the list to pairs of elements
                history_messages = [
                    [messages[k], messages[k + 1]] for k in range(0, len(messages), 2)
                ]
                if not system_prompt:
                    system_prompt = system_prompts[i] if system_prompts else ""
                yield self.prompt_template.get_dialog(
                    history_messages, system_prompt=system_prompt
                )

        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = self.tokenizer.encode(
                    text=dialog[2 * i], add_special_tokens=(i == 0)
                )
                target_ids = self.tokenizer.encode(
                    text=dialog[2 * i + 1], add_special_tokens=False
                )

                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(self.max_length * (len(source_ids) / total_len))
                max_target_len = int(self.max_length * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len - 1:  # eos token
                    target_ids = target_ids[: max_target_len - 1]
                if len(source_ids) > 0 and source_ids[0] == self.tokenizer.eos_token_id:
                    source_ids = source_ids[1:]
                if (
                    len(target_ids) > 0
                    and target_ids[-1] == self.tokenizer.eos_token_id
                ):
                    target_ids = target_ids[:-1]
                if (
                    len(input_ids) + len(source_ids) + len(target_ids) + 1
                    > self.max_length
                ):
                    break

                input_ids += (
                    source_ids + target_ids + [self.tokenizer.eos_token_id]
                )  # add eos token for each turn
                if self.script_args.train_on_inputs:
                    labels += source_ids + target_ids + [self.tokenizer.eos_token_id]
                else:
                    labels += (
                        [self.IGNORE_INDEX] * len(source_ids)
                        + target_ids
                        + [self.tokenizer.eos_token_id]
                    )

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    def filter_empty_labels(self, example):
        """Remove empty labels dataset."""
        return not all(label == self.IGNORE_INDEX for label in example["labels"])

    def load_dataset(self):# -> tuple[Dataset | Any | None, Dataset | Any | None]:
        train_dataset = None
        max_train_samples = 0
        if self.training_args.do_train:
            if "train" not in self.raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = self.raw_datasets["train"].shuffle(seed=42)
            max_train_samples = len(train_dataset)
            if (
                self.data_args.max_train_samples is not None
                and self.data_args.max_train_samples > 0
            ):
                max_train_samples = min(
                    len(train_dataset), self.data_args.max_train_samples
                )
                train_dataset = train_dataset.select(range(max_train_samples))
            logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
            with self.training_args.main_process_first(
                desc="Train dataset tokenization"
            ):
                train_dataset = train_dataset.shuffle().map(
                    self.preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=train_dataset.column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
                train_dataset = train_dataset.filter(
                    self.filter_empty_labels,
                    num_proc=self.data_args.preprocessing_num_workers,
                )
                logger.debug(f"Num train_samples: {len(train_dataset)}")
                logger.debug("Tokenized training example:")
                logger.debug(
                    f"Decode input_ids[0]:\n{self.tokenizer.decode(train_dataset[0]['input_ids'])}"
                )
                replaced_labels = [
                    label if label != self.IGNORE_INDEX else self.tokenizer.pad_token_id
                    for label in list(train_dataset[0]["labels"])
                ]
                logger.debug(
                    f"Decode labels[0]:\n{self.tokenizer.decode(replaced_labels)}"
                )

        eval_dataset = None
        max_eval_samples = 0
        if self.training_args.do_eval:
            with self.training_args.main_process_first(
                desc="Eval dataset tokenization"
            ):
                if "validation" not in self.raw_datasets:
                    raise ValueError("--do_eval requires a validation dataset")
                eval_dataset = self.raw_datasets["validation"]
                max_eval_samples = len(eval_dataset)
                if (
                    self.data_args.max_eval_samples is not None
                    and self.data_args.max_eval_samples > 0
                ):
                    max_eval_samples = min(
                        len(eval_dataset), self.data_args.max_eval_samples
                    )
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                eval_size = len(eval_dataset)
                logger.debug(f"Num eval_samples: {eval_size}")
                if eval_size > 500:
                    logger.warning(
                        f"Num eval_samples is large: {eval_size}, "
                        f"training slow, consider reduce it by `--max_eval_samples=50`"
                    )
                logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
                eval_dataset = eval_dataset.map(
                    self.preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    remove_columns=eval_dataset.column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )
                eval_dataset = eval_dataset.filter(
                    self.filter_empty_labels,
                    num_proc=self.data_args.preprocessing_num_workers,
                )
                logger.debug(f"Num eval_samples: {len(eval_dataset)}")
                logger.debug("Tokenized eval example:")
                logger.debug(self.tokenizer.decode(eval_dataset[0]["input_ids"]))
        
        return train_dataset, eval_dataset


if __name__ == "__main__":
    from arguments import DataArguments, ModelArguments, ScriptArguments
    from transformers import AutoTokenizer, HfArgumentParser, Seq2SeqTrainingArguments

    parser = HfArgumentParser((ModelArguments, DataArguments, ScriptArguments, Seq2SeqTrainingArguments))
    model_args, data_args, script_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    load_data = dataset(model_args=model_args, data_args=data_args, script_args=script_args, training_args=training_args, tokenizer=tokenizer)
    train_dataset, eval_dataset = load_data.load_dataset()
