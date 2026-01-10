from src.io import load_story, load_backstory, load_dataset, save_result, calc_accuracy
from src.pipeline import run_pipeline
import os

def main():
    #story = load_story("data/story_001.txt")
    #backstory = load_backstory("data/backstory_001.txt")
    if os.path.exists("outputs/results.csv"):
        os.remove("outputs/results.csv")

    if os.path.exists("data/train.csv"):
        df=load_dataset("data/train.csv")
    else:
        raise FileNotFoundError("train.csv")
    
    story_cache = {}
    
    for row in df.itertuples(index=False):
        bkname = row.book_name.strip().lower()
        if bkname not in story_cache:
            if os.path.exists(f"data/story/{bkname}.txt"):
                story_cache[bkname] = load_story(f"data/story/{bkname}.txt")
            else:
                raise FileNotFoundError(bkname)
        story = story_cache[bkname]
        backstory=row.content
        label, rationale = run_pipeline(story, backstory)
        save_result(row.id,label,rationale)

    print("Accuracy: ",calc_accuracy("data/train.csv","outputs/results.csv"))

    #print("Prediction:", label)
    #print("Rationale:", rationale)

if __name__ == "__main__":
    main()
