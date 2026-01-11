from transformers import pipeline


# Load NLI model once
_nli = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    return_all_scores=True,
    truncation=True,
    max_length=512
)


def judge_consistency(claim, retrieved_chunks):
    """
    Judge whether a single claim is consistent with retrieved evidence.

    Returns:
        label (int): 1 = consistent, 0 = contradiction
        rationale (str)
        confidence (float)
    """

    if not retrieved_chunks:
        return 1, "No relevant evidence found.", 0.5

    # Join retrieved chunks directly (they are STRINGS)
    evidence_text = " ".join(retrieved_chunks)

    # NLI input format
    premise = evidence_text[:3000]  # safety cap
    hypothesis = claim

    outputs = _nli(f"{premise} </s></s> {hypothesis}")

    # HuggingFace output format handling
    scores = {item["label"].lower(): item["score"] for item in outputs[0]}

    contradiction_score = scores.get("contradiction", 0.0)
    entailment_score = scores.get("entailment", 0.0)

    if contradiction_score > 0.7:
        return 0, "Evidence contradicts the claim.", contradiction_score
    else:
        return 1, "No strong contradiction detected.", max(entailment_score, 1 - contradiction_score)
