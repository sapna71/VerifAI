from src.chunk import chunk_text

def read_novel(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    novel_text = read_novel("data/story_001.txt")
    chunks = chunk_text(novel_text)

    print("Total characters:", len(novel_text))
    print("Total chunks:", len(chunks))

    if chunks:
        print("Sample chunk ID:", chunks[0]["id"])
        print("Sample chunk text:\n", chunks[0]["text"][:300])
    else:
        print("No chunks created — check input text.")

if __name__ == "__main__":
    main()
