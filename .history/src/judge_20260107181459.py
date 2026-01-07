# src/judge.py

"""
Judging module for verifying consistency between
character backstory and retrieved story evidence.
"""

from transformers import pipeline


def judge_consistency(backstory_text, retrieved_chunks):
    """
    Judge whether the story is consistent with the backstory.

    Args:
        backstory_text (str): Character backstory
        retrieved_chunks (list): List of retrieved chunk dicts

    Returns:
        tuple:
            label (int): 1 if consistent, 0 if inconsistent
            rationale (str): Explanation of the decision
    """

    if not backstory_text or not retrieved_chunks:
        return 0, "Insufficient information to judge consistency."

    # Combine evidence text
    evidence_text = " ".join(chunk["text"] for chunk in retrieved_chunks)

    # Load NLI model
    classifier = pipeline(
        "text-classification",
        model="roberta-large-mnli",
        device=-1  # CPU
    )

    # Hypothesis-style prompt (NLI formulation)
    input_text = (
        f"Premise: {evidence_text}\n"
        f"Hypothesis: {backstory_text}"
    )

    # Run classification
    outputs = classifier(input_text, top_k=None)

    # Convert output to dict
    scores = {item["label"]: item["score"] for item in outputs[0]}

    # Labels: ENTAILMENT / NEUTRAL / CONTRADICTION
    entail_score = scores.get("ENTAILMENT", 0)
    contradict_score = scores.get("CONTRADICTION", 0)

    if entail_score >= contradict_score:
        label = 1
        rationale = (
            "The retrieved story evidence supports the backstory. "
            "Key events and character traits are consistent."
        )
    else:
        label = 0
        rationale = (
            "The retrieved story evidence contradicts the backstory. "
            "Some events or character traits are inconsistent."
        )

    return label, rationale
