from src.io import load_story, load_backstory
from src.pipeline import run_pipeline

def main():
    story = load_story("data/story_001.txt")
    backstory = load_backstory("data/backstory_001.txt")

    label, rationale = run_pipeline(story, backstory)

    print("Prediction:", label)
    if rationale:
        print("Rationale:", rationale)

if __name__ == "__main__":
    main()
