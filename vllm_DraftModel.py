from vllm import LLM, SamplingParams
import time

prompts = ["What color is the sky?", "A dog is pet known for?", "How does one make pizza?"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=4,
    speculative_config={
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,        
        "method": "draft_model",
},
)

start = time.time()
outputs = specdec_model.generate(prompts, sampling_params)
end = time.time()

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("Model took:  ", end-start)