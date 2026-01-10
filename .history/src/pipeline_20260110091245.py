from .retrieve import retrieve_chunks
from .judge import judge_consistency


def split_into_claims(backstory_text):
    """
    Split backstory into simple sentence-level claims.
    Baseline approach: split by period.
    """
    return [c.strip() for c in backstory_text.split('.') if c.strip()]


def run_pipeline(story_text, backstory_text):
    """
    Main orchestration pipeline:
    - Split backstory into claims
    - Retrieve evidence per claim
    - Judge each claim
    - Aggregate into final decision
    """

    claims = split_into_claims(backstory_text)

    claim_results = []
    rationales = []

    for claim in claims:
        # Retrieve relevant chunks for this claim
        retrieved_chunks = retrieve_chunks(story_text, claim)

        # Judge consistency of this claim
        label, rationale, confidence = judge_consistency(claim, retrieved_chunks)

        claim_results.append({
            "claim": claim,
            "label": label,
            "confidence": confidence
        })

        rationales.append(
            f"Claim: '{claim}' → {'consistent' if label == 1 else 'contradiction'} "
            f"(confidence: {confidence:.2f})"
        )

    # Aggregation logic:
    # If ANY claim is a strong contradiction → overall contradiction
    final_label = 1
    for result in claim_results:
        if result["label"] == 0 and result["confidence"] >= 0.7:
            final_label = 0
            break

    final_rationale = " | ".join(rationales)

    return final_label, final_rationale
