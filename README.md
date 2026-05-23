# Causal-GPT-10.7M
10.7M-parameter decoder-only transformer trained from scratch on Tiny Shakespeare. Features an optimized Causal KV-Caching inference engine reducing generation runtime complexity to $O(T)$, dynamic shape tracking to eliminate asynchronous cuBLAS memory faults, mixed-precision acceleration, and a robust out-of-vocabulary validation guardrail.
