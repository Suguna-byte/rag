"""
Retrieval evaluation: compare BM25 vs dense vs hybrid (RRF) on a small golden set.
Metrics: Recall@1, Recall@3, MRR. A hit = the retrieved chunk's section heading matches the expected one.
Questions are deliberately paraphrased so pure keyword matching is not enough.
"""
from rag import HybridRAG

GOLDEN = [
    ("How do I merge rankings from a keyword search and a vector search?", "Fusion: Reciprocal Rank Fusion (RRF)"),
    ("Which retrieval approach handles synonyms and paraphrases?", "Sparse vs dense retrieval"),
    ("Why rescore the candidate list with a cross-encoder?", "Reranking"),
    ("Explain the knobs k1 and b in the classic keyword ranking formula", "BM25 in one breath"),
    ("What are HNSW and IVF?", "Vector databases and indexes"),
    ("How can I measure whether my retriever finds the right passages?", "Evaluating RAG"),
    ("The model makes up confident false facts. How do I reduce that?", "Hallucination"),
    ("Cheap way to adapt a big model without updating all weights", "Fine-tuning and PEFT"),
    ("What is HyDE and multi-query?", "Query transformations"),
    ("Malicious text inside a retrieved web page tells the model to ignore its rules", "Security: prompt injection"),
    ("Why is the attention score divided by a square root?", "Why divide by sqrt(d_k)"),
    ("A user double-clicked submit and two records were saved", "Operforce AI — likely questions"),
    ("How should I talk about the three-month job and the gap?", "Timeline questions to prepare for"),
    ("One-hot encoding nucleotides for the bacteria promoter project", "E. coli Detection Using DNA Promoter Classification"),
    ("Precision versus recall and when accuracy misleads", "Classification metrics"),
    ("What should I ask the panel at the end?", "Questions to ask the interviewer"),
    ("Design a question answering assistant over company documents", 'System design: "Design a document Q&A assistant for a company"'),
    ("Should I fine-tune or retrieve when knowledge changes weekly?", "RAG vs fine-tuning vs long context"),
    ("How do I log tokens, latency and cost per request in production?", "LLMOps and observability"),
    ("Small pieces vs big pieces when splitting documents", "Chunking strategies"),
]


def evaluate(rag: HybridRAG, mode: str, k: int = 3):
    r1 = r3 = mrr = 0.0
    misses = []
    for q, expected in GOLDEN:
        headings = [c.heading for c, _ in rag.retrieve(q, k=10, mode=mode)]
        rank = headings.index(expected) + 1 if expected in headings else None
        r1 += rank == 1
        r3 += rank is not None and rank <= k
        mrr += 1 / rank if rank else 0
        if not rank or rank > k:
            misses.append(q)
    n = len(GOLDEN)
    return r1 / n, r3 / n, mrr / n, misses


if __name__ == "__main__":
    rag = HybridRAG()
    print(f"{len(GOLDEN)} golden questions | {len(rag.chunks)} chunks | dense={rag.dense.kind}\n")
    print(f"{'mode':<8}{'Recall@1':>10}{'Recall@3':>10}{'MRR':>8}")
    all_misses = {}
    for mode in ("bm25", "dense", "hybrid"):
        r1, r3, mrr, misses = evaluate(rag, mode)
        all_misses[mode] = misses
        print(f"{mode:<8}{r1:>10.2f}{r3:>10.2f}{mrr:>8.2f}")
    print("\nHybrid misses (not in top 3):")
    for q in all_misses["hybrid"] or ["none"]:
        print(" -", q)
