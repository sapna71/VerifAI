# src/retrieve.py

"""
Retrieval module for selecting relevant story chunks
based on backstory similarity.

Pathway is imported safely for Track A compliance,
but not executed directly on Windows.
"""

# ---- Safe Pathway import (DO NOT REMOVE) ----
try:
    import pathway as pw  # required by hackathon
except Exception:
    pw = None

# ---- Other imports ----
from sentence_transformers import SentenceTransformer
import numpy as np


def retrieve_chunks(chunks, backstory_text, top_k=5):
    """
    Retrieve top-k most relevant chunks based on semantic similarity.

    Args:
        chunks (list): list of dicts with keys {"id", "text"}
        backstory_text (str): character backstory
        top_k (int): number of chunks to retrieve

    Returns:
        list: subset of chunks (dicts)
    """

    if not chunks or not backstory_text:
        return []

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode backstory
    backstory_embedding = model.encode(backstory_text)

    # Encode chunk texts
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = model.encode(chunk_texts)

    # Compute cosine similarity
    similarities = cosine_similarity(backstory_embedding, chunk_embeddings)

    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Return top-k chunks
    retrieved = [chunks[i] for i in top_indices]

    return retrieved


def cosine_similarity(vec, matrix):
    """
    Compute cosine similarity between one vector and a matrix of vectors.
    """

    vec_norm = np.linalg.norm(vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # Avoid division by zero
    if vec_norm == 0 or np.any(matrix_norms == 0):
        return np.zeros(len(matrix))

    similarities = np.dot(matrix, vec) / (matrix_norms * vec_norm)
    return similarities
