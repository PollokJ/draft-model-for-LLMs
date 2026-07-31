from vllm import LLM, SamplingParams

# VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 python vllm_DraftModel.py
# Simply running the 32B model with the 0.6B or 8B Draftmodels without any training stays within +/- 10% of the original performance.
# Noteworthy that this is a standard draft model and not eagle3 algo

prompts = [
    "What color is the sky?",
    "A dog is pet known for?",
    "How does one make pizza?",
    "The roman empire fell because",
    "What is the capital of France?",
    "Why is the sky blue?",
    "Explain gravity in simple terms.",
    "Who wrote Hamlet?",
    "What causes earthquakes?",
    "Write a short story about a lost astronaut.",
    "Tell me a joke.",
    "What is 17 multiplied by 24?",
    "If all cats are animals and all animals are mortal, then",
    "How do airplanes stay in the air?",
    "The future of artificial intelligence will",
]

#{
#  "model": "Qwen/Qwen3-32B",
#  "messages": [
#    {
#      "role": "user",
#      "content": "Tell me a joke about animals."
#    }
#  ],
#  "temperature": 0.8,
#  "max_tokens": 300
#}

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200, min_tokens=50)

llm = LLM(
    model="Qwen/Qwen3-32B",
    tensor_parallel_size=4,
    max_model_len = 8192,
    max_num_seqs = 2,
    gpu_memory_utilization=0.9,
    enforce_eager=True,
    speculative_config={
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 3,        
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

del llm