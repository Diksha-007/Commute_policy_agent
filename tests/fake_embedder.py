"""A small, deterministic embedder for tests, so the suite does not depend on any embedding model.

It hashes words into a fixed number of dimensions (a "bag of words"), drops filler words and the plural "s", and
normalises. Two texts that share meaningful words get a high cosine similarity, texts that share none get 0.
That is enough for the tests, which check what the CODE does with retrieved chunks (isolation, versioning, citations,
failures), not how well a model understands language. No download, no network, same result on every machine.

The real model is exercised by a few opt-in tests: `pytest -m real_model` (tests/test_real_embedder.py).
"""
import re
import zlib

import numpy as np

DIM = 768
STOP = {
    "a", "an", "the", "is", "are", "was", "be", "do", "does", "did", "i", "me", "my", "we", "you", "your", "it", "its",
    "of", "to", "in", "on", "at", "for", "and", "or", "if", "with", "by", "as", "than", "that", "this", "will", "can",
    "what", "who", "how", "get", "have", "has", "any", "so", "not", "no", "up", "from", "then", "when", "which",
}
_WORD = re.compile(r"[a-z0-9₹]+")


def tokens(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]                        # "guards" == "guard"
        out.append(w)
    return out


class HashingEmbedder:
    dim = DIM

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for w in tokens(text):
            v[zlib.crc32(w.encode("utf-8")) % DIM] += 1.0     # crc32, not hash(): stable across runs and machines
        n = np.linalg.norm(v)
        return v / n if n else v

    def documents(self, texts):
        return np.stack([self._vector(t) for t in texts]) if len(texts) else np.zeros((0, DIM), dtype=np.float32)

    queries = documents          # symmetric: no query instruction
