# LLM Fundamentals

## Transformer architecture
A transformer processes tokens with stacked layers of self-attention and feed-forward networks, with residual connections and layer normalization. Self-attention lets every token look at every other token: each token produces a query, a key and a value; attention weights are softmax(QK^T / sqrt(d_k)), and the output is the weighted sum of values. Multi-head attention runs several attention operations in parallel so the model can track different relationships. Decoder-only models such as GPT, Claude and Llama use causal masking so a token only sees earlier tokens.

## Why divide by sqrt(d_k)
Dot products grow with dimension, which pushes softmax into saturated regions with tiny gradients. Scaling keeps the values in a stable range.

## Tokenization
Text is split into subword tokens using algorithms like BPE (byte-pair encoding). A token is roughly 3 to 4 characters of English. Token counts drive cost, latency and context limits. Non-English text and code often use more tokens per word.

## Positional information
Attention alone ignores order, so models add positional information. Modern models mostly use rotary position embeddings (RoPE), which also help extend context length.

## Training stages
1. Pretraining: next-token prediction on huge text corpora; learns language and world knowledge.
2. Supervised fine-tuning (SFT): trained on instruction and response examples to follow instructions.
3. Preference tuning: RLHF (reward model plus reinforcement learning such as PPO) or DPO (directly optimizes on preferred vs rejected pairs, simpler and more stable). Constitutional AI uses principles and AI feedback.

## Decoding and sampling parameters
- Temperature: scales logits. Low (0 to 0.3) for factual, deterministic tasks like extraction and RAG; higher for creative writing.
- Top-p (nucleus): sample from the smallest set of tokens whose probability sums to p.
- Top-k: sample from the k most likely tokens.
- Max tokens caps output length; stop sequences end generation early.

## Context window and KV cache
The context window is the maximum number of tokens the model can attend to. During generation the model caches the keys and values of previous tokens (the KV cache) so it does not recompute them; this makes generation faster but uses memory proportional to context length. Prompt caching at the API level reuses a repeated prompt prefix to cut cost and latency.

## Hallucination
The model produces fluent but false content because it predicts plausible text, not verified facts. Reduce it with grounding (RAG), asking for citations, letting the model say it does not know, lower temperature, structured outputs, and post-hoc verification.

## Fine-tuning and PEFT
- Full fine-tuning updates all weights: expensive and needs lots of GPU memory.
- LoRA freezes the base model and learns small low-rank matrices added to weight matrices (W + BA, where B and A have small rank r). Few trainable parameters, cheap to store, swappable adapters.
- QLoRA applies LoRA on a 4-bit quantized base model so large models fit on a single GPU.
- When to fine-tune: consistent format or tone, a narrow classification or extraction task, reducing prompt length, or distilling a large model into a small one. Not the first tool for adding facts.

## Quantization
Storing weights in lower precision (8-bit, 4-bit) reduces memory and speeds up inference with a small quality loss. Formats you might hear: GPTQ, AWQ, GGUF (used by llama.cpp and Ollama).

## Embedding models vs generative models
Embedding models output a fixed vector for a text and are used for search, clustering and classification. Generative models output text. They are different models and you choose them separately in a RAG system.

## Model selection trade-offs
Quality, latency, cost per token, context length, data privacy (API vs self-hosted), and license. A common pattern is routing: a small fast model for simple requests and a larger model for hard ones.
