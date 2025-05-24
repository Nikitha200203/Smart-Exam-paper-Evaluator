import torch
from transformers import pipeline

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

pipe = pipeline("text-generation", model=MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

messages = [
    {"role": "user", "content": "What is 2+2?"},
]

result = pipe(messages)
print(result)
