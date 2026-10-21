# RAG Systems — Core Interview Topic

## What RAG is and why it exists
Retrieval-Augmented Generation retrieves relevant passages from your own data at query time and gives them to the language model as context. It solves three problems: the model does not know your private or recent data, it hallucinates when it lacks facts, and fine-tuning is expensive and hard to update. With RAG, updating knowledge means re-indexing documents, not retraining.

## The two pipelines
Indexing (offline): load documents, clean them, split into chunks, embed each chunk, store vectors plus metadata in an index.
Querying (online): take the user question, optionally rewrite it, retrieve the top-k chunks, optionally rerank, build a prompt with the chunks, generate the answer with citations.

## Chunking strategies
- Fixed-size with overlap: simple; overlap prevents a fact being cut in half at a boundary.
- Structure-aware: split on headings, paragraphs, or markdown sections first, then by size. Usually better because each chunk stays on one topic.
- Semantic chunking: split where embedding similarity between adjacent sentences drops.
- Parent-child (small-to-big): retrieve small precise chunks, but send the larger parent section to the model for context.
- Trade-off: small chunks give precise retrieval but lose context; large chunks keep context but dilute the embedding and waste tokens.

## Embeddings
An embedding model maps text to a vector so that similar meanings land close together. Similarity is usually cosine similarity, which is the dot product of normalized vectors. Common choices: OpenAI text-embedding-3, Cohere embed, open-source BGE, E5, GTE and Nomic models. Choose by language support, domain, dimension size and cost. Always embed queries and documents with the same model.

## Sparse vs dense retrieval
- Sparse (BM25, TF-IDF): keyword matching. Strong on exact terms, IDs, names, acronyms and rare words. No training needed.
- Dense (embeddings): matches meaning and paraphrases ("car" vs "automobile"). Weak on exact codes and rare terms.
- Hybrid search runs both and fuses the results. It is the production default because each covers the other's blind spots.

## BM25 in one breath
BM25 scores a document for a query by summing, over query terms, IDF(term) times a saturated term frequency. The parameter k1 (about 1.2 to 2.0) controls how fast repeated terms stop adding score, and b (about 0.75) controls length normalization so long documents are not unfairly favoured.

## Fusion: Reciprocal Rank Fusion (RRF)
RRF combines ranked lists without needing comparable scores: score(d) = sum over lists of 1 / (k + rank(d)), with k typically 60. It works because BM25 scores and cosine scores live on different scales, and ranks do not.

## Reranking
A cross-encoder reranker (for example Cohere Rerank or bge-reranker) reads the query and each candidate together and scores relevance. It is far more accurate than a bi-encoder but too slow to run over the whole corpus, so you retrieve 50 candidates cheaply and rerank them to the best 5.

## Query transformations
- Query rewriting: turn a chatty follow-up ("what about last year?") into a standalone question using chat history.
- Multi-query: generate several phrasings and merge results.
- HyDE: have the model write a hypothetical answer, embed that, and search with it.
- Decomposition: split a multi-part question into sub-questions.

## Vector databases and indexes
- Options: pgvector (inside PostgreSQL), FAISS (library), Chroma, Qdrant, Weaviate, Pinecone, Milvus, Elasticsearch/OpenSearch.
- Exact search compares the query to every vector: accurate but O(n).
- Approximate nearest neighbour (ANN) indexes trade a little recall for big speed: HNSW (graph-based, fast, memory heavy, the common default) and IVF (cluster vectors, search only nearby clusters; often combined with product quantization to compress).
- Metadata filtering (by year, region, user, permissions) is essential in real systems.

## Prompting for grounded answers
Tell the model to answer only from the provided context, cite chunk IDs, and say "I don't know" when the context lacks the answer. Put the instructions clearly, keep the context ordered by relevance, and avoid stuffing too many chunks, since models can miss information in the middle of long contexts.

## Evaluating RAG
Evaluate retrieval and generation separately.
- Retrieval metrics: recall@k (did the right chunk appear in the top k), precision@k, MRR (mean reciprocal rank of the first correct hit), nDCG.
- Generation metrics: faithfulness or groundedness (is every claim supported by the context), answer relevance, correctness against a reference answer.
- Tools: RAGAS, TruLens, DeepEval, and LLM-as-judge with a clear rubric. Calibrate the judge against some human labels.
- Build a golden set of 50 to 200 real questions with expected sources before tuning anything.

## Common RAG failure modes and fixes
- Right answer not retrieved: improve chunking, add hybrid search, add query rewriting, tune k.
- Retrieved but ignored: rerank, reduce the number of chunks, put the best chunk first.
- Hallucination: stricter prompt, require citations, verify claims, return "not found".
- Stale data: incremental re-indexing with document hashes.
- Tables and PDFs: parse tables separately, keep headers with rows.
- Security: enforce document-level permissions at retrieval time, not in the prompt.

## RAG vs fine-tuning vs long context
- RAG: for knowledge that changes or is private; gives citations.
- Fine-tuning: for behaviour, style, format or a narrow skill, not for injecting facts.
- Long context: fine for a few documents, but cost and latency grow with tokens and recall degrades in the middle; RAG still wins for large corpora.
- They combine: a fine-tuned model can sit on top of a RAG pipeline.
