from transformers import pipeline
import torch

print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))

# For GPU use device=0, for CPU use device=-1
pipe = pipeline("text-generation", model="cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct", device=0 if torch.cuda.is_available() else -1)

# Use a simple string prompt and print the result
resp = pipe("Who are you?", max_new_tokens=200)
print(resp)