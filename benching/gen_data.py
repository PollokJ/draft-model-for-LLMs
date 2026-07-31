from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os
import re

model = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(model)

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

def clean(t):
    t = re.sub(r"\[\d+\]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def build_dataset(target_len=500, stride=500, max_items=500):
    out = []

    for doc in ds:
        text = clean(doc["text"])
        ids = tok.encode(text, truncation=False)

        for i in range(0, len(ids) - target_len, stride):
            chunk = ids[i:i+target_len]
            if len(chunk) < target_len:
                continue

            out.append({
                "prompt": tok.decode(chunk, skip_special_tokens=True),
                "output_len": 256
            })

            if len(out) >= max_items:
                return out

    return out


def save_dataset(target_len, stride):
    data = build_dataset(target_len=target_len, stride=stride)

    filename = f"Inputs_{target_len}_stride_{stride}.jsonl"

    with open(filename, "w") as f:
        for x in data:
            f.write(json.dumps(x) + "\n")

    print("saved:", filename, "items:", len(data))



save_dataset(5000, 250)
save_dataset(10000, 250)
