"""Embedding model shared by chunking (ingest time) and retrieval (query time).

Production model: BAAI/bge-base-en-v1.5, run locally through fastembed (ONNX). It was chosen in experiment E1 and
confirmed on a 4x larger question set in E3/E4 (about +0.07 facts found over MiniLM, interval [+0.03, +0.12]).
It needs no API quota, no network at query time and no PyTorch, and it is deterministic.

bge models are asymmetric: a QUESTION is embedded with an instruction prefix, a DOCUMENT (a policy chunk, or a HyDE
passage written in policy style) is embedded without one. `embed` is for documents, `embed_query` for questions.

The model files are downloaded into the Docker image at build time (see Dockerfile).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable

import numpy as np

# Anything that maps a list of texts to an (n, dim) array of L2-normalised vectors.
EmbedFn = Callable[[list[str]], np.ndarray]

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DIMENSIONS = 768
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
CACHE_DIR = os.environ.get("FASTEMBED_CACHE_PATH", "/opt/fastembed")


class BgeEmbedder:
    """The production embedder: bge-base through fastembed. Loads the model on first use."""

    dim = DIMENSIONS

    @property
    def _model(self):
        return _load_bge()

    def documents(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.embed(list(texts), batch_size=32))

    def queries(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.embed([QUERY_INSTRUCTION + t for t in texts], batch_size=32))


@lru_cache(maxsize=1)
def _load_bge():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=MODEL_NAME, cache_dir=CACHE_DIR)


# The embedder every function below uses. Production never changes it. Tests install a small deterministic stand-in
# (tests/fake_embedder.py) so they do not depend on any embedding model; a few opt-in tests use the real one.
_embedder = None


def set_embedder(embedder) -> None:
    """Install an object with .documents(texts) and .queries(texts) returning L2-normalised arrays; None restores bge."""
    global _embedder
    _embedder = embedder


def _active():
    return _embedder if _embedder is not None else _BGE


_BGE = BgeEmbedder()


def _unit(vectors) -> np.ndarray:
    vecs = np.asarray(list(vectors), dtype=np.float32)
    if vecs.size == 0:
        return np.zeros((0, DIMENSIONS), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed(texts: list[str]) -> np.ndarray:
    """Embed DOCUMENTS (policy chunks, sentences, HyDE passages), L2-normalised so a dot product is cosine similarity."""
    if not texts:
        return np.zeros((0, DIMENSIONS), dtype=np.float32)
    return _active().documents(list(texts))


def embed_query(texts: list[str]) -> np.ndarray:
    """Embed QUESTIONS (bge adds its query instruction)."""
    if not texts:
        return np.zeros((0, DIMENSIONS), dtype=np.float32)
    return _active().queries(list(texts))


class DocumentEmbeddingFunction:
    """Chroma embedding function for the DOCUMENT side. (Questions are embedded by the store with `embed_query` and
    passed to Chroma as vectors, because Chroma's own query path cannot add an instruction prefix.)"""

    def __call__(self, input):  # noqa: A002 (Chroma requires this exact parameter name)
        return embed(list(input)).tolist()


@lru_cache(maxsize=1)
def get_embedding_function() -> DocumentEmbeddingFunction:
    return DocumentEmbeddingFunction()


# ---- Experiments only -------------------------------------------------------------------------------
# The earlier model (all-MiniLM-L6-v2 through Chroma's default ONNX function). Experiments E1-E4 chunked their long policies
# with it, so it stays available to reproduce them; the app does not use it.

@lru_cache(maxsize=1)
def _minilm():
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()


def minilm_embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vecs = np.asarray(_minilm()(texts), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms
