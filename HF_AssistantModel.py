import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model = "Qwen/Qwen3-32B"
draft_model_name  = "Qwen/Qwen3-8B"

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

inputs = tokenizer("Tell me how to groom a dog.", return_tensors="pt").to("cuda")


# TODO: Add warmup for better testing/benching

# Normal model output
torch.cuda.synchronize()
start = time.time()

out = og_model.generate(
    **inputs,
    max_new_tokens=500,
    min_new_tokens=500,
    do_sample=True,
    temperature=0.7,
    use_cache=True
)

torch.cuda.synchronize()
end = time.time()
og_time = end - start

print()
print("OG Model:  ", end - start)
print(tokenizer.decode(out[0]))
print()

# SpecDec output
torch.cuda.synchronize()
start = time.time()

out = og_model.generate(
    **inputs,
    assistant_model=specdec_model,
    max_new_tokens=500,
    min_new_tokens=500,
    do_sample=True,
    temperature=0.7,
    use_cache=True,
    num_assistant_tokens = 4
    # assistant_tokenizer=tokenizer Same for both models rn so this line is not really needed
)

torch.cuda.synchronize()
end = time.time()

specdec_time = end - start
print()
print("SpecDec Model:  ", specdec_time)
print(tokenizer.decode(out[0]))
print()

print("We had a speedup of: ", og_time/specdec_time)