from transformers import pipeline

nli = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    return_all_scores=True
)

def contradiction_score(premise, hypothesis):
    """
    Returns probability of contradiction
    """
    results = nli(f"{premise} </s></s> {hypothesis}")[0]
    for r in results:
        if r["label"].lower() == "contradiction":
            return r["score"]
    return 0.0


def split_into_claims(backstory):
    return [s.strip() for s in backstory.split(";") if s.strip()]


def judge(backstory, retrieved_chunks, contradiction_threshold=0.6):
    claims = split_into_claims(backstory)

    violations = []

    for claim in claims:
        for chunk in retrieved_chunks:
            if claim not in chunk["matched_claims"]:
                continue

            score = contradiction_score(chunk["text"], claim)

            if score >= contradiction_threshold:
                violations.append({
                    "claim": claim,
                    "chunk_id": chunk["chunk_id"],
                    "score": score
                })

    
    if len(violations) > 0:
        return 0  

    return 1  