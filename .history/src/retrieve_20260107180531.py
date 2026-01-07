import numpy as np
try:
    import pathway as pw
except Exception:
    pw = None

from sentence_transformers import SentenceTransformer
from collections import defaultdict

def split_into_claims(backstory: str):
    return [s.strip() for s in backstory.split(";") if s.strip()]


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class Retriever:
    def __init__(self, chunk_table: pw.Table, embed_model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model_name)

        self.df = chunk_table.to_pandas()

        self.df["embedding"] = list(
            self.model.encode(self.df["text"].tolist(), show_progress_bar=True)
        )

    def retrieve(self, backstory, character, k_per_claim=5):
        claims = split_into_claims(backstory)
        retrieved = defaultdict(lambda: {
            "chunk_id": None,
            "text": None,
            "score": 0.0,
            "matched_claims": []
        })

        for claim in claims:
            query = f"{character}. {claim}"
            q_emb = self.model.encode(query)

            self.df["score"] = self.df["embedding"].apply(
                lambda e: cosine_sim(q_emb, e)
            )

            topk = self.df.sort_values("score", ascending=False).head(k_per_claim)

            for _, row in topk.iterrows():
                cid = row["id"]
                if retrieved[cid]["chunk_id"] is None:
                    retrieved[cid]["chunk_id"] = cid
                    retrieved[cid]["text"] = row["text"]
                    retrieved[cid]["score"] = row["score"]

                retrieved[cid]["matched_claims"].append(claim)
                retrieved[cid]["score"] = max(retrieved[cid]["score"], row["score"])

        return list(retrieved.values())
