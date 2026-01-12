
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks.
    Returns a list of dicts with id and text.
    """
    chunks = []
    start = 0
    chunk_id = 0

    if not text or text.strip() == "":
        return chunks

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
