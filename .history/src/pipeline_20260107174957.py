from src.chunk import chunk_text
from src.retrieve import retrieve_chunks
from src.judge import judge_consistency

def run_pipeline(story_text, backstory_text):
    # 1. Chunk the story
    chunks = chunk_text(story_text)

    if not chunks:
        return 0, "No story content to evaluate"

    # 2. Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(chunks, backstory_text)

    if not retrieved_chunks:
        return 0, "No relevant evidence retrieved"

    # 3. Judge consistency
    label, rationale = judge_consistency(backstory_text, retrieved_chunks)

    return label, rationale
