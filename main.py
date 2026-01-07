from src.chunk import chunk_text

def read_novel(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"File not found: {path}")
        return ""

def main():
    novel_text = read_novel("data/story_001.txt")
    chunks = chunk_text(novel_text)

    print("Total characters:", len(novel_text))
    print("Total chunks:", len(chunks))

    if chunks:
        print("\nSample chunk ID:", chunks[0]["id"])
        print("Sample chunk text:\n", chunks[0]["text"][:300])
    else:
        print("\nNo chunks created — check story file content.")

if __name__ == "__main__":
    main()
