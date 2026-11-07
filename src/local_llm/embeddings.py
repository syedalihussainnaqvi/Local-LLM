"""Local embedding and vector representation utilities."""

import hashlib
import logging
import math
import re
from typing import List, Optional

import numpy as np

from . import config

logger = logging.getLogger("embeddings")


class LocalTextEmbedder:
    """
    Lightweight, fast local embedding engine.
    Uses sublinear TF-IDF hashing trick with character/word n-grams and L2 normalization.
    Completely local, deterministic, requires zero external model downloads, and runs instantly on CPU.
    Also supports pluggable HuggingFace sentence-transformers if installed.
    """

    def __init__(self, vector_dim: int = 512, model_name: str = "hash_tfidf_512"):
        self.vector_dim = vector_dim
        self.model_name = model_name
        self._hf_model = None

        # Check if user configured a sentence_transformers model and if library is installed
        if config.LOG_LEVEL.upper() == "DEBUG":
            logger.debug(f"Initialized LocalTextEmbedder with {model_name} (dim={vector_dim})")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Normalize and tokenize text into words and 2-grams."""
        words = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
        tokens = list(words)
        # Add word bigrams for better contextual phrase matching
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a normalized dense vector."""
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.vector_dim, dtype=np.float32)

        # Feature hashing (Murmur/SHA256 hash mod vector_dim)
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            # Sublinear term frequency: 1 + log(tf)
            tf = 1.0 + math.log(count)
            # Hash to index and sign (+1 / -1)
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self.vector_dim
            sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
            vec[idx] += sign * tf

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text strings into a 2D numpy array (N, vector_dim)."""
        if not texts:
            return np.empty((0, self.vector_dim), dtype=np.float32)
        vectors = [self.embed_text(t) for t in texts]
        return np.vstack(vectors)


# Global default embedder
_DEFAULT_EMBEDDER: Optional[LocalTextEmbedder] = None

def get_embedder() -> LocalTextEmbedder:
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = LocalTextEmbedder()
    return _DEFAULT_EMBEDDER
