# app/embedding.py

from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
from .config import EMBEDDING_MODEL


# ============================================================
# Load Model Once (Singleton Pattern)
# ============================================================

@lru_cache(maxsize=1)
def get_model():
    """
    Loads the embedding model once.
    Cached so it is not reloaded repeatedly.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


# ============================================================
# Embed Text Function
# ============================================================

def embed_text(texts, batch_size=128):
    """
    Converts list of texts into normalized embeddings.
    """

    model = get_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return embeddings.astype(np.float32)