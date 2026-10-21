# Interview-Prep RAG

A hybrid-search RAG system over your own interview notes (`kb/`). Ask it questions while you study; explain how it works in the interview.

## Run it
```bash
pip install numpy scikit-learn
python rag.py "How does reciprocal rank fusion work?"
python rag.py                 # interactive
python evaluate.py            # retrieval metrics: BM25 vs dense vs hybrid
```
Optional upgrades (picked up automatically):
```bash
pip install sentence-transformers   # real embeddings: BAAI/bge-small-en-v1.5
pip install anthropic && export ANTHROPIC_API_KEY=...   # Claude writes the answer with citations
```

## Architecture
```
kb/*.md ──► chunk by "## " heading (220 words, 40 overlap, title+heading prepended)
                │
     ┌──────────┴───────────┐
   BM25 (from scratch)   Dense: bge-small  |  fallback LSA (TF-IDF → SVD)
     └──────────┬───────────┘
        Reciprocal Rank Fusion (k=60) → top 4 chunks
                │
   Answer: Claude (grounded prompt, cite ids, "I don't know")  |  fallback extractive
```

## Design decisions (say these out loud)
1. **Structure-aware chunking.** Split on headings so each chunk is one topic; window long sections with overlap so facts aren't cut at boundaries. The doc title and heading are prepended to each chunk (cheap contextual chunking) so "Reranking" text still knows it's about RAG.
2. **BM25 written by hand.** Exact terms, acronyms (HNSW, LoRA, RRF) and names. k1=1.5 saturates repeated terms; b=0.75 normalizes length.
3. **Dense retrieval** catches paraphrases. Vectors are unit-normalized, so the dot product is cosine similarity. In the sandbox it was built in, no model download was possible, so it falls back to LSA (latent semantic analysis) — weaker, but the interface is identical.
4. **RRF instead of score averaging.** BM25 scores are unbounded, cosines are in [-1, 1]; ranks are comparable, scores aren't.
5. **Grounded generation.** Prompt says answer only from chunks, cite ids, admit when the answer is missing; temperature 0. Without an LLM, an extractive answerer picks the bullets with the most IDF-weighted query overlap and drops anything below half the best score.
6. **Evaluation before tuning.** 20 paraphrased golden questions with expected sections; Recall@1, Recall@3, MRR per mode.

## Results (offline LSA mode)
| mode   | Recall@1 | Recall@3 | MRR  |
|--------|---------:|---------:|-----:|
| BM25   | 0.70 | 0.80 | 0.76 |
| Dense  | 0.70 | 0.75 | 0.76 |
| Hybrid | 0.75 | 0.80 | 0.79 |

Hybrid wins on rank quality. Remaining misses are heavy paraphrases ("double-clicked submit" vs "duplicate submissions") — exactly what a neural embedding model or a cross-encoder reranker fixes.

## What I'd change for production
- Neural embeddings + a cross-encoder reranker over the top 20–50.
- A real vector store (pgvector/Qdrant with HNSW) and metadata filters.
- Query rewriting with chat history for follow-ups.
- Faithfulness evaluation (RAGAS or LLM-as-judge) on top of retrieval metrics, run in CI.
- Tracing of prompts, chunks, tokens, latency and cost; incremental re-indexing by content hash.
