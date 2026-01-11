from sentence_transformers import SentenceTransformer, util
import numpy as np


# Load embedding model once
_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_story_index(story_text):
    words = story_text.split()
    chunk_size = 400
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

    # Embed
    chunk_embeddings = _model.encode(chunks, convert_to_tensor=True)
    return chunks, chunk_embeddings


def retrieve_chunks(query, chunks, chunk_embeddings, top_k=5):
    """
    Simple retrieval:
    - Split story into chunks
    - Embed chunks
    - Return top-k most similar chunks (as strings)
    """

    query_embedding = _model.encode(query, convert_to_tensor=True)

    # Similarity
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = scores.topk(k=min(top_k, len(chunks))).indices.tolist()

    # Return TEXT ONLY
    return [chunks[i] for i in top_indices]
