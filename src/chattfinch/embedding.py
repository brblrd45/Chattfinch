"""The embedding model: loaded once, used for chunk sizing, indexing and queries."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from chattfinch import config


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    """Tokens in `text` by the model's tokenizer, excluding the special tokens
    that the model adds around every input."""
    return len(get_model().tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])


def embed(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Unit-length embeddings, one row per text (so dot product = cosine similarity)."""
    return get_model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              show_progress_bar=show_progress, convert_to_numpy=True)
