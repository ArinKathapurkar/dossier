"""Embedding generation with an MPS correctness guard.

Model: BAAI/bge-small-en-v1.5 (384-d). Small enough to encode the whole corpus on an
Apple-Silicon laptop in minutes, strong enough on financial prose to be a reasonable
first-stage retriever; the cross-encoder reranker recovers precision afterwards.

The MPS parity guard exists because PyTorch's Metal backend has historically produced
silently *wrong* transformer output on some kernels -- not an exception, just different
numbers. A wrong embedding table would look like a bad retriever, which is exactly the
kind of failure that gets misdiagnosed as a modelling problem. So we encode a sample on
both devices and compare before trusting MPS.
"""

from __future__ import annotations

import numpy as np

from ..config import get_config

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_model(device: str | None = None, model_name: str | None = None):
    from sentence_transformers import SentenceTransformer

    cfg = get_config()
    name = model_name or cfg.embed_model
    device = device or pick_device()
    key = (name, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SentenceTransformer(name, device=device)
    return _MODEL_CACHE[key]


def mps_parity_check(sample_texts: list[str], threshold: float = 0.999) -> dict:
    """Encode the same sample on MPS and CPU; require mean cosine >= threshold.

    Returns a report; the caller decides whether to demote to CPU.
    """
    device = pick_device()
    if device != "mps":
        return {"ran": False, "device": device, "reason": "mps unavailable", "ok": True}
    sample = sample_texts[:32]
    if not sample:
        return {"ran": False, "device": device, "reason": "no sample", "ok": True}
    mps = np.asarray(load_model("mps").encode(sample, normalize_embeddings=True, show_progress_bar=False))
    cpu = np.asarray(load_model("cpu").encode(sample, normalize_embeddings=True, show_progress_bar=False))
    cos = float(np.mean(np.sum(mps * cpu, axis=1)))
    return {
        "ran": True,
        "device": device,
        "mean_cosine": round(cos, 6),
        "threshold": threshold,
        "ok": cos >= threshold,
    }


class Embedder:
    """Encodes passages and queries. Queries get the model's instruction prefix; passages
    do not -- bge asymmetric retrieval expects exactly that split."""

    def __init__(self, device: str | None = None, model_name: str | None = None):
        cfg = get_config()
        self.cfg = cfg
        self.model_name = model_name or cfg.embed_model
        requested = device or pick_device()
        self.parity = mps_parity_check(
            ["dossier embedding parity probe sentence number %d about revenue and cash flow." % i for i in range(32)]
        )
        if requested == "mps" and self.parity.get("ran") and not self.parity.get("ok"):
            requested = "cpu"
            self.parity["demoted_to_cpu"] = True
        self.device = requested
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.device, self.model_name)
        return self._model

    def encode_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size or self.cfg.embed_batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_passages([self.cfg.query_prefix + query])[0]
