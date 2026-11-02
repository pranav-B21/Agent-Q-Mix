"""Text-embedding task features, projected to the controller's fixed task-feature size."""

from functools import lru_cache
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model():
    # sentence_transformers imports HF datasets, which this repo's datasets/ package shadows.
    from datasets.hf_loader import _swap_in_hf, _swap_back
    removed_paths, local_modules = _swap_in_hf()
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME, device="cpu")
    finally:
        _swap_back(removed_paths, local_modules)


@lru_cache(maxsize=1)
def _projection(in_dim, out_dim):
    # Seeded, so every process and every run uses the same projection.
    return np.random.default_rng(0).standard_normal((in_dim, out_dim)) / np.sqrt(out_dim)


@lru_cache(maxsize=4096)
def _encode(task, dim):
    emb = _model().encode(task, normalize_embeddings=True)
    feat = emb @ _projection(len(emb), dim)
    return feat / max(np.linalg.norm(feat), 1e-8)


def encode_task(task, dim=32):
    return _encode(task, dim).copy()
