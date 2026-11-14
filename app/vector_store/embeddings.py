import abc
import hashlib
import re
import numpy as np
from app.core.config import settings


def normalize_arabic_for_embedding(text: str) -> str:
    """Standardizes Arabic characters and strips diacritics for embedding consistency."""
    if not text:
        return ""
    # Remove tashkeel / diacritics
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    # Normalize Alef variants
    text = re.sub(r"[إأآٱ]", "ا", text)
    # Normalize Taa Marbuta
    text = re.sub(r"ة", "ه", text)
    # Normalize Yaa
    text = re.sub(r"ى", "ي", text)
    # Clean whitespace and lowercase latin
    text = text.lower().strip()
    return text


class BaseEmbeddingEngine(abc.ABC):
    """Abstract interface for dense vector embedding engines."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        pass

    @abc.abstractmethod
    def embed_text(self, text: str) -> list[float]:
        pass

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class LocalSemanticEmbeddingEngine(BaseEmbeddingEngine):
    """
    Lightweight, deterministic, 100% offline Arabic & multilingual embedding engine.
    Uses subword n-gram hashing and Gaussian random projection to 384 dimensions,
    yielding high-speed vector representations compatible with Qdrant Cosine distance.
    Zero external dependencies beyond NumPy, zero cost, completely local.
    """

    DIMENSION: int = 384

    def __init__(self, seed: int = 42):
        self._dim = self.DIMENSION
        # Initialize deterministic projection matrix (10000 hash buckets -> 384 dense dims)
        rng = np.random.RandomState(seed)
        self.num_buckets = 8192
        self.projection_matrix = rng.randn(self.num_buckets, self._dim).astype(np.float32)
        # Normalize projection matrix
        self.projection_matrix /= np.linalg.norm(self.projection_matrix, axis=1, keepdims=True)

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_token(self, token: str) -> int:
        """Deterministically hashes a string token to a bucket index."""
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
        return h % self.num_buckets

    def embed_text(self, text: str) -> list[float]:
        """Computes a 384-dimensional L2-normalized dense embedding for the text."""
        norm_text = normalize_arabic_for_embedding(text)
        if not norm_text:
            return [0.0] * self._dim

        words = norm_text.split()
        features = {}

        # 1. Whole words
        for w in words:
            features[w] = features.get(w, 0.0) + 1.5

        # 2. Character 3-grams and 4-grams (captures Arabic root morphology)
        for w in words:
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    tri = w[i : i + 3]
                    features[tri] = features.get(tri, 0.0) + 0.8
            if len(w) >= 4:
                for i in range(len(w) - 3):
                    four = w[i : i + 4]
                    features[four] = features.get(four, 0.0) + 0.5

        # Project features into dense vector
        dense_vec = np.zeros(self._dim, dtype=np.float32)
        for feat, weight in features.items():
            bucket = self._hash_token(feat)
            dense_vec += weight * self.projection_matrix[bucket]

        # L2 Unit Normalization (so dot product equals cosine similarity)
        norm = np.linalg.norm(dense_vec)
        if norm > 1e-9:
            dense_vec = dense_vec / norm
        else:
            dense_vec = np.zeros(self._dim, dtype=np.float32)

        return [float(x) for x in dense_vec]


_embedding_instance: BaseEmbeddingEngine | None = None


def get_embedding_engine() -> BaseEmbeddingEngine:
    """Returns singleton instance of the configured embedding engine."""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = LocalSemanticEmbeddingEngine()
    return _embedding_instance
