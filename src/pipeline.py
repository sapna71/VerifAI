# src/pipeline.py

def run_pipeline(story_text, backstory_text):
    """
    Full pipeline:
    1. Chunk story
    2. Retrieve relevant chunks
    3. Judge consistency
    4. Return label + rationale
    """

    # Step 1: chunking
    chunks = None  # will be filled

    # Step 2: retrieval
    retrieved_chunks = None  # friend

    # Step 3: judging
    label = None  # friend
    rationale = None  # optional

    return label, rationale
