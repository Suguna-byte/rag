"""
Web UI for the interview-prep RAG. Standard library only (no Flask needed).

    python app.py            -> open http://localhost:8000
    python app.py 9000       -> custom port

Endpoints
  GET  /            the page (index.html in this folder)
  GET  /api/info    index stats
  POST /api/ask     {"q": "...", "mode": "hybrid|bm25|dense", "k": 4}
  GET  /api/eval    runs the golden-set evaluation from evaluate.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from rag import HybridRAG, rrf, tokenize

HERE = Path(__file__).parent
POOL = 20
print("Building index...", flush=True)
RAG = HybridRAG()
GENERATOR = "Claude" if os.getenv("ANTHROPIC_API_KEY") else "extractive"


def ask(q: str, mode: str = "hybrid", k: int = 4) -> dict:
    t0 = time.perf_counter()
    sparse = RAG.bm25.scores(tokenize(q))
    dense = RAG.dense.scores(q)
    sparse_rank = [int(i) for i in np.argsort(-sparse)[:POOL] if sparse[i] > 0]
    dense_rank = [int(i) for i in np.argsort(-dense)[:POOL]]
    fused = rrf([sparse_rank, dense_rank])
    fused_rank = sorted(fused, key=fused.get, reverse=True)
    order = {"bm25": sparse_rank, "dense": dense_rank}.get(mode, fused_rank)[:k]
    t_retrieve = time.perf_counter() - t0

    chunks = [RAG.chunks[i] for i in order]
    answer, generator = None, "extractive"
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            answer, generator = RAG._llm_answer(q, chunks), "Claude"
        except Exception as e:
            print(f"[LLM unavailable: {e}]", file=sys.stderr)
    if answer is None:
        answer = RAG._extractive_answer(q, chunks)
    t_total = time.perf_counter() - t0

    def pos(lst, i):
        return lst.index(i) + 1 if i in lst else None

    hits = []
    for i in order:
        c = RAG.chunks[i]
        hits.append({
            "id": c.id, "source": c.source, "heading": c.heading,
            "text": c.text.split(": ", 1)[-1],
            "bm25_rank": pos(sparse_rank, i), "dense_rank": pos(dense_rank, i),
            "fused_rank": pos(fused_rank, i),
            "bm25_score": round(float(sparse[i]), 3), "dense_score": round(float(dense[i]), 3),
            "rrf_score": round(fused.get(i, 0.0), 5),
        })
    return {
        "query": q, "mode": mode, "answer": answer, "generator": generator,
        "terms": sorted(set(tokenize(q))), "hits": hits,
        "ms": {"retrieve": round(t_retrieve * 1000, 1), "total": round(t_total * 1000, 1)},
    }


def run_eval() -> dict:
    from evaluate import GOLDEN, evaluate
    rows = []
    for mode in ("bm25", "dense", "hybrid"):
        r1, r3, mrr, misses = evaluate(RAG, mode)
        rows.append({"mode": mode, "r1": r1, "r3": r3, "mrr": mrr, "misses": misses})
    return {"n": len(GOLDEN), "rows": rows}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            sources = sorted({c.source for c in RAG.chunks})
            self._json({"chunks": len(RAG.chunks), "docs": len(sources), "dense": RAG.dense.kind,
                        "generator": GENERATOR})
        elif self.path == "/api/eval":
            self._json(run_eval())
        else:
            self._json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path != "/api/ask":
            return self._json({"error": "Not found"}, 404)
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            q = str(data.get("q", "")).strip()
            if not q:
                return self._json({"error": "Type a question first."}, 400)
            mode = data.get("mode", "hybrid")
            k = max(1, min(8, int(data.get("k", 4))))
            self._json(ask(q, mode, k))
        except Exception as e:
            self._json({"error": f"Search failed: {e}"}, 500)

    def log_message(self, fmt, *args):  # quieter console
        pass


if __name__ == "__main__":
    port = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8000)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"Ready: {len(RAG.chunks)} chunks | dense={RAG.dense.kind} | answers={GENERATOR}")
    print(f"Open {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
