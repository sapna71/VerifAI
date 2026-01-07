# src/chunk.py

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks and assigns IDs.
    """
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append({
            "id": chunk_id,
            "text": chunk
        })

        chunk_id += 1
        start += chunk_size - overlap

    return chunks
