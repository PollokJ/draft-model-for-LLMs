import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

model = "Qwen/Qwen3-8B"
draft_model_name  = "Qwen/Qwen3-1.7B"

# Rn this is using the same tokenizer for both models. Both from Qwen 3, so probably the same. Might be an issue with other models tho
tokenizer = AutoTokenizer.from_pretrained(model)

og_model = AutoModelForCausalLM.from_pretrained(
    model,
    dtype=torch.float16,
    device_map="auto",
)

specdec_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name,
    dtype=torch.float16,
    device_map="auto",
)

inputs = tokenizer("Tell me what color the sky is.", return_tensors="pt").to("cuda")

# Normal model output
torch.cuda.synchronize()
start = time.time()

out = og_model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

torch.cuda.synchronize()
end = time.time()

print("OG Model:  ", end - start)


# SpecDec output
torch.cuda.synchronize()
start = time.time()

out = specdec_model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    assistant_model=specdec_model,
    assistant_tokenizer=tokenizer #Same for both models rn
)

torch.cuda.synchronize()
end = time.time()

print("SpecDec Model:  ", end - start)