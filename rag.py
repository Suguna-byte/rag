"""
Interview-prep RAG: hybrid retrieval (BM25 + dense) fused with RRF, then a grounded answer.

Runs fully offline with scikit-learn. Upgrades itself automatically when available:
  - sentence-transformers installed  -> real neural embeddings (BAAI/bge-small-en-v1.5)
  - ANTHROPIC_API_KEY set + `anthropic` -> Claude writes the final answer from retrieved chunks
Otherwise it falls back to LSA embeddings and an extractive (sentence-selection) answer.

Usage:
  python rag.py "How does reciprocal rank fusion work?"
  python rag.py            # interactive mode
"""
from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

KB_DIR = Path(__file__).parent / "kb"
STOPWORDS = set("""a an the and or of to in on for is are was were be been it its this that with as by at
from how what why when which who do does did can could should would i you your we our they their me my
about into than then so if not no vs""".split())


# ---------------------------------------------------------------- 1. Load + chunk
@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    text: str
    meta: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if t not in STOPWORDS]


def chunk_markdown(path: Path, max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    """Structure-aware chunking: split on '## ' headings, then window long sections with overlap."""
    raw = path.read_text(encoding="utf-8")
    doc_title = raw.splitlines()[0].lstrip("# ").strip()
    sections = re.split(r"\n(?=## )", raw)
    chunks: list[Chunk] = []
    for sec in sections:
        lines = sec.strip().splitlines()
        if not lines or lines[0].startswith("# "):  # skip doc title block
            lines = lines[1:]
            if not lines:
                continue
        heading = lines[0].lstrip("# ").strip() if lines[0].startswith("##") else doc_title
        body_lines = [l.strip() for l in (lines[1:] if lines[0].startswith("##") else lines) if l.strip()]
        # Keep line breaks (bullets) inside the chunk so the answer step can split on them.
        words = "\n".join(body_lines).replace("\n", " \n ").split(" ")
        words = [w for w in words if w]
        step = max_words - overlap
        for start in range(0, max(len(words), 1), step):
            piece = " ".join(words[start:start + max_words]).replace(" \n ", "\n")
            if not piece:
                break
            chunks.append(Chunk(
                id=f"{path.stem}#{len(chunks)}",
                source=path.name,
                heading=heading,
                # Prepend doc title + heading: cheap "contextual chunking" so a chunk knows where it came from.
                text=f"{doc_title} > {heading}: {piece}",
            ))
            if start + max_words >= len(words):
                break
    return chunks


def load_corpus(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks = []
    for p in sorted(kb_dir.glob("*.md")):
        chunks.extend(chunk_markdown(p))
    return chunks


# ---------------------------------------------------------------- 2. Sparse retriever (BM25, from scratch)
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = self.lens.mean()
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        # BM25 IDF with +1 so it never goes negative for very common terms
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, tf in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
            for t in query:
                if t in tf:
                    f = tf[t]
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return out


# ---------------------------------------------------------------- 3. Dense retriever
class DenseRetriever:
    """Neural embeddings if sentence-transformers is installed, else LSA (TF-IDF -> SVD)."""

    def __init__(self, texts: list[str]):
        self.kind = "lsa"
        try:
            from sentence_transformers import SentenceTransformer  # optional
            self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            self.matrix = self.model.encode(texts, normalize_embeddings=True)
            self.kind = "bge-small-en-v1.5"
            return
        except Exception:
            pass
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        tfidf = self.vec.fit_transform(texts)
        k = max(2, min(64, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        self.matrix = normalize(self.svd.fit_transform(tfidf))

    def embed_query(self, q: str) -> np.ndarray:
        if self.kind != "lsa":
            return self.model.encode([q], normalize_embeddings=True)[0]
        return normalize(self.svd.transform(self.vec.transform([q])))[0]

    def scores(self, q: str) -> np.ndarray:
        return self.matrix @ self.embed_query(q)  # cosine similarity (vectors are unit length)


# ---------------------------------------------------------------- 4. Hybrid + RRF
def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRAG:
    def __init__(self, kb_dir: Path = KB_DIR):
        self.chunks = load_corpus(kb_dir)
        texts = [c.text for c in self.chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.dense = DenseRetriever(texts)

    def retrieve(self, query: str, k: int = 4, mode: str = "hybrid", pool: int = 20):
        sparse = self.bm25.scores(tokenize(query))
        dense = self.dense.scores(query)
        sparse_rank = [i for i in np.argsort(-sparse)[:pool] if sparse[i] > 0]
        dense_rank = list(np.argsort(-dense)[:pool])
        if mode == "bm25":
            order = sparse_rank
        elif mode == "dense":
            order = dense_rank
        else:
            fused = rrf([sparse_rank, dense_rank])
            order = sorted(fused, key=fused.get, reverse=True)
        return [(self.chunks[i], i) for i in order[:k]]

    # ------------------------------------------------------------ 5. Generation
    def answer(self, query: str, k: int = 4) -> tuple[str, list[Chunk]]:
        hits = [c for c, _ in self.retrieve(query, k=k)]
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                return self._llm_answer(query, hits), hits
            except Exception as e:  # fall back rather than crash the demo
                print(f"[LLM unavailable: {e}; using extractive answer]", file=sys.stderr)
        return self._extractive_answer(query, hits), hits

    def _llm_answer(self, query: str, hits: list[Chunk]) -> str:
        import anthropic
        context = "\n\n".join(f"<chunk id='{c.id}'>\n{c.text}\n</chunk>" for c in hits)
        prompt = (
            "Answer the question using ONLY the chunks below. Cite chunk ids in square brackets. "
            "If the chunks do not contain the answer, say you don't know.\n\n"
            f"{context}\n\nQuestion: {query}"
        )
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.getenv("RAG_MODEL", "claude-sonnet-4-5"), max_tokens=600, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _extractive_answer(self, query: str, hits: list[Chunk], n_sent: int = 4) -> str:
        """No LLM: pick the sentences from retrieved chunks that best match the query, keep citations."""
        q = set(tokenize(query))
        idf = self.bm25.idf
        cands = []
        for rank, c in enumerate(hits):
            body = c.text.split(": ", 1)[-1]
            # Unit = one bullet (keeps "question? Answer: ..." pairs together); long paragraphs -> sentences.
            units = []
            for line in body.split("\n"):
                line = re.sub(r"^[-\d.\s]+", "", line).strip()
                units.extend([line] if len(line.split()) <= 70 else re.split(r"(?<=[.!?])\s+", line))
            for s in units:
                toks = set(tokenize(s))
                hit = q & toks
                if len(toks) < 4 or not hit:
                    continue  # no query-term overlap -> not evidence
                overlap = sum(idf.get(t, 0) for t in hit) / (sum(idf.get(t, 1) for t in q) + 1e-9)
                score = overlap + 0.08 * (len(hits) - rank)  # prefer higher-ranked chunks
                if s.endswith("?"):
                    score -= 0.3  # a question restated is not an answer
                cands.append((score, s, c.id))
        cands.sort(key=lambda x: -x[0])
        best = cands[0][0] if cands else 0
        seen, lines = set(), []
        for score, s, cid in cands:
            if s in seen or score < 0.5 * best:
                continue
            seen.add(s)
            lines.append(f"- {s} [{cid}]")
            if len(lines) == n_sent:
                break
        return "\n".join(lines) if lines else "I don't know — nothing relevant in the knowledge base."


def main():
    rag = HybridRAG()
    gen = "Claude" if os.getenv("ANTHROPIC_API_KEY") else "extractive"
    print(f"Indexed {len(rag.chunks)} chunks | dense={rag.dense.kind} | sparse=BM25 | fusion=RRF | answer={gen}\n")
    queries = sys.argv[1:] or iter(lambda: input("Ask (blank to quit)> ").strip(), "")
    for q in queries:
        ans, hits = rag.answer(q)
        print(f"Q: {q}\n{ans}\nSources: " + ", ".join(f"{h.id} ({h.heading})" for h in hits) + "\n")


if __name__ == "__main__":
    main()
