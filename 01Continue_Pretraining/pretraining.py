import math
import os
from loguru import logger
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    BloomForCausalLM,
    BloomTokenizerFast,
    HfArgumentParser,
    LlamaForCausalLM,
    Seq2SeqTrainingArguments,
    Trainer,
    is_torch_tpu_available,
    set_seed,
)
try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:  # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled

from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer import TRAINING_ARGS_NAME
from sklearn.metrics import accuracy_score

from arguments import DataArguments, ModelArguments, ScriptArguments
from data import dataset, fault_tolerance_data_collator

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "llama": (AutoConfig, LlamaForCausalLM, AutoTokenizer),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}

parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))
model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()

logger.info(f"Model args: {model_args}")
logger.info(f"Data args: {data_args}")
logger.info(f"Training args: {training_args}")
logger.info(f"Script args: {script_args}")
logger.info(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)

# Set seed before initializing model.
set_seed(training_args.seed)

# Load tokenizer
config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]

tokenizer_kwargs = {
    "cache_dir": model_args.cache_dir,
    "use_fast": model_args.use_fast_tokenizer,
    "trust_remote_code": model_args.trust_remote_code,
}
tokenizer_name_or_path = model_args.tokenizer_name_or_path
if not tokenizer_name_or_path:
    tokenizer_name_or_path = model_args.model_name_or_path
tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

dataset = dataset(model_args=model_args, data_args=data_args, tokenizer=tokenizer, training_args=training_args)
# 如果存在processor_data文件，则使用processor处理数据集
if os.path.exists("processed_data"):
    from datasets import load_from_disk
    lm_datasets = load_from_disk("processed_data")
else:
    lm_datasets = dataset.load_data()
fault_tolerance_data_collator = fault_tolerance_data_collator()


train_dataset = None
max_train_samples = 0
if training_args.do_train:
    if "train" not in lm_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = lm_datasets['train']
    max_train_samples = len(train_dataset)
    if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    logger.debug(f"Num train_samples: {len(train_dataset)}")
    logger.debug("Tokenized training example:")
    logger.debug(tokenizer.decode(train_dataset[0]['input_ids']))

eval_dataset = None
max_eval_samples = 0
if training_args.do_eval:
    if "validation" not in lm_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = lm_datasets["validation"]
    max_eval_samples = len(eval_dataset)
    if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    logger.debug(f"Num eval_samples: {len(eval_dataset)}")
    logger.debug("Tokenized eval example:")
    logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

# Load model
if model_args.model_type and model_args.model_name_or_path:
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1  # 如果全局变量WORLD_SIZE不等于1，则ddp为True，进行分布式数据并行训练
    if ddp:
        model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
        logger.warning("FSDP and DeepSpeed ZeRO-3 are both currently incompatible with QLoRA.")

    config_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }
    config = config_class.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    load_in_4bit = model_args.load_in_4bit
    load_in_8bit = model_args.load_in_8bit
    if load_in_4bit and load_in_8bit:
        raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")
    elif load_in_8bit or load_in_4bit:
        logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
        if load_in_8bit:
            config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            if script_args.qlora:
                config_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            else:
                config_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        device_map= model_args.device_map,
        **config_kwargs,
    )
else:
    raise ValueError(f"Error, model_name_or_path is None, Continue PT must be loaded from a pre-trained model")

def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

if script_args.use_peft:
    logger.info("Fine-tuning method: LoRA(PEFT)")
    if script_args.peft_path is not None:
        logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
        model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
    else:
        logger.info("Init new peft model")
        if load_in_8bit or load_in_4bit:
            model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)
        target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
        if target_modules and 'all' in target_modules:
            target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)
        modules_to_save = script_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
            # Resize the embedding layer to match the new tokenizer
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Peft target_modules: {target_modules}")
        logger.info(f"Peft lora_rank: {script_args.lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=script_args.lora_rank,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            modules_to_save=modules_to_save)
        model = get_peft_model(model, peft_config)
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)
    model.print_trainable_parameters()
else:
    logger.info("Fine-tuning method: Full parameters training")
    model = model.float()
    print_trainable_parameters(model)

# Initialize our Trainer
if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
else:
    model.config.use_cache = True
model.enable_input_require_grads()
if not ddp and torch.cuda.device_count() > 1:
    # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)

def accuracy(predictions, references, normalize=True, sample_weight=None):
    return {
        "accuracy": float(accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight))
    }

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics, we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

trainer = SavePeftModelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=fault_tolerance_data_collator,
    compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
    if training_args.do_eval and not is_torch_tpu_available()
    else None,
)

def save_model_zero3(model, tokenizer, args, trainer):
    """Save the model for deepspeed zero3.
    refer https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L209
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)

def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Training
if training_args.do_train:
    logger.info("*** Train ***")
    logger.debug(f"Train dataloader example: {next(iter(trainer.get_train_dataloader()))}")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = max_train_samples
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    model.config.use_cache = True  # enable cache after training
    tokenizer.padding_side = "left"  # restore padding side
    tokenizer.init_kwargs["padding_side"] = "left"

    if trainer.is_world_process_zero():
        logger.debug(f"Training metrics: {metrics}")
        logger.info(f"Saving model checkpoint to {training_args.output_dir}")
        if is_deepspeed_zero3_enabled():
            save_model_zero3(model, tokenizer, training_args, trainer)
        else:
            save_model(model, tokenizer, training_args)

# Evaluation
if training_args.do_eval:
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    metrics["eval_samples"] = max_eval_samples
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    if trainer.is_world_process_zero():
        logger.debug(f"Eval metrics: {metrics}")