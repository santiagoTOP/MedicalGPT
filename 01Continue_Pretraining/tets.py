from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "./models/qwen/Qwen2___5-0___5B",
    trust_remote_code=True,
)
print(tokenizer("你好吗？"))
print(tokenizer.tokenize("你好吗？"))