from vllm import LLM, SamplingParams
import time
import torch
import re
import requests

prompts = [
    #"What color is the sky?",
    #"A dog is pet known for?",
    #"How does one make pizza?",
    "What is the result of 2+2 - (14+8)?"]


sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=4,
    speculative_config={
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,        
        "method": "draft_model",
    }
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print()
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print()



def get_metric(name, text):
    match = re.search(rf"{name}\s+([0-9e\+\-\.]+)", text)
    return float(match.group(1)) if match else 0.0

metrics_text = requests.get("http://localhost:8000/metrics").text

accepted = get_metric("vllm:spec_decode_num_accepted_tokens_total", metrics_text)
draft = get_metric("vllm:spec_decode_num_draft_tokens_total", metrics_text)

if draft > 0:
    print("Acceptance rate:", accepted / draft)
else:
    print("No draft tokens yet")