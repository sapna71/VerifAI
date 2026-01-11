from sentence_transformers import SentenceTransformer, util
import numpy as np
from src.store import chunks_to_table


# Load embedding model once
_model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def build_story_index(story_text):
    words = story_text.split()
    chunk_size = 400
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

    # Embed
    chunk_embeddings = _model.encode(chunks, convert_to_tensor=True)
    chunk_data = [
        {"chunk_id": i, "text": c, "embedding": emb.tolist()}
        for i, (c, emb) in enumerate(zip(chunks, chunk_embeddings))
    ]

    pw_table = chunks_to_table(chunk_data)
    return pw_table


def retrieve_chunks(query, pw_table, top_k=5):
    """
    Simple retrieval:
    - Split story into chunks
    - Embed chunks
    - Return top-k most similar chunks (as strings)
    """

    query_emb = _model.encode(query)
    df = pw_table.to_pandas() 

    df["score"] = df["embedding"].apply(
        lambda e: cosine_sim(np.array(e), query_emb)
    )

    # Similarity
    top_df = df.sort_values("score", ascending=False).head(top_k)
    retrieved_texts = top_df["text"].tolist()

    return retrieved_texts
