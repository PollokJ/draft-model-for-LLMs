# Draft Model for LLMs

Large Language Models (LLMs) process text in two stages: prefill, where the
input is read, followed by decode, where tokens are generated. Empirical results showed that a compressed version of DeepSeek R1 can run on our clusters but achieves only moderate decoding speeds (~10 tokens/sec), while prefilling is about 7× faster. Speculative decoding is a promising technique to
approach prefilling speeds during generation: a small, lightweight draft
model generates candidate tokens that are then verified by the “main model”.

Because entire sequences of draft tokens can be verified at once, this approach has the potential to achieve near prefilling throughput even during the
decoding phase. However, preliminary work has uncovered only a single
available draft model (the baseline), with a further caveat that said model was
designed for the uncompressed version of DeepSeek R1.

In this thesis, the student will create draft models for at least two models, one
of which is the compressed version of DeepSeek R1. Specifically, the student
should design and implement a pipeline that produces draft models which
have measurable improvement in decoding speed over the baseline when deployed in comparable settings (e.g. on SafeAI). Further proxy metrics can be
the average token acceptance rate and average length of draft sequence before
rejection.
Optional extensions include but are not limited to testing the pipeline for different hardware settings (e.g. Avalon 1/2, TCML), measuring impact of compression (e.g. quantization) on the draft and/or main model, or doing a deeper
dive into sophisticated methods such as PARD.