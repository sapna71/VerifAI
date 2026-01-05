def read_novel(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    novel_text = read_novel("data/story_001.txt")
    print("Number of characters in novel:", len(novel_text))

if __name__ == "__main__":
    main()
