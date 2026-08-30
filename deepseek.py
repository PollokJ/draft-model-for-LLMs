import json
import os
from huggingface_hub import snapshot_download
from llama_cpp import Llama
from datasets import load_dataset

print("Downloading/Verifying DeepSeek GGUF shards")
model_dir = snapshot_download(
    repo_id="unsloth/DeepSeek-R1-GGUF",
    allow_patterns=["*UD-IQ1_S*"],
)

model_path = os.path.join(
    model_dir,
    "DeepSeek-R1-UD-IQ1_S",
    "DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf",
)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Could not find model shard at: {model_path}")

print(f"Loading model from: {model_path}")
llm = Llama(
    model_path=model_path,
    n_gpu_layers=99,
    n_ctx=4096,
    logits_all=True,
    embedding=True,
)

print("Loading dataset")
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:10]")

output_file_path = os.path.expanduser("~/.cache/deepseek_teacher_dataset.jsonl")

with open(output_file_path, "w") as outfile:
    counter = 0
    for item in dataset:
        print(counter)
        counter = counter + 1
        messages = item["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

        if user_msg:
            prompt = f"<｜User｜>{user_msg}<｜Assistant｜>"

            output = llm(
                prompt,
                max_tokens=128,
                temperature=0.7,
                stop=["<｜end_of_sentence｜>"]
            )

            assistant_response = output["choices"][0]["text"]

            record = {
                "text": prompt + assistant_response,
                "hidden_states": []
            }
            outfile.write(json.dumps(record) + "\n")

print("Generation completde")
