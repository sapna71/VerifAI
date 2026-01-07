# src/judge.py

"""
Judging module for verifying consistency between
character backstory and retrieved story evidence.
"""

from transformers import pipeline


def judge_consistency(backstory_text, retrieved_chunks):
    """
    Judge whether the story is consistent with the backstory.

    Returns:
        label (int): 1 if consistent, 0 if inconsistent
        rationale (str)
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

    # NLI-style input
    input_text = (
        f"Premise: {evidence_text}\n"
        f"Hypothesis: {backstory_text}"
    )

    outputs = classifier(input_text)

    # ---- SAFE OUTPUT HANDLING ----
    # outputs is usually: [{'label': 'ENTAILMENT', 'score': 0.87}]
    if isinstance(outputs, list) and isinstance(outputs[0], dict):
        label_name = outputs[0]["label"]
        score = outputs[0]["score"]
    else:
        return 0, "Unexpected model output format."

    # Decision logic
    if label_name == "ENTAILMENT":
        label = 1
        rationale = (
            f"The story is consistent with the backstory "
            f"(confidence: {score:.2f})."
        )
    elif label_name == "CONTRADICTION":
        label = 0
        rationale = (
            f"The story contradicts the backstory "
            f"(confidence: {score:.2f})."
        )
    else:  # NEUTRAL
        label = 1
        rationale = (
            f"The story neither contradicts nor strongly confirms "
            f"the backstory (confidence: {score:.2f})."
        )

    return label, rationale
