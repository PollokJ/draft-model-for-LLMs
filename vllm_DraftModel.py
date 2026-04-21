from vllm import LLM, SamplingParams
import time

og_model = LLM(model ="Qwen/Qwen3-8B")

specDec_models = LLM(
    model = "Qwen/Qwen3-8B",
    speculative_model = "Qwen/Qwen3-1.7B"
)

parameters = SamplingParams(
    max_tokens = 100,
    temperature=0.7
)

start = time.time()
out = og_model.generate("Write a story about a robot", parameters)
end = time.time()
print("OG Model took:  ", end-start)

start = time.time()
out = specDec_models.generate("Write a story about a robot", parameters)
end = time.time()
print("SpecDec Model took:  ", end-start)