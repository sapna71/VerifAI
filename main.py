from src.io import load_story, load_backstory, load_dataset, save_result, calc_accuracy
from src.pipeline import run_pipeline
from src.retrieve import build_story_index
import os

def main():
    
    if os.path.exists("outputs/results.csv"):
        os.remove("outputs/results.csv")

    if os.path.exists("data/test.csv"):
        df=load_dataset("data/test.csv")
    else:
        raise FileNotFoundError("test.csv")
    
    story_cache = {}
    
    for row in df.itertuples(index=False):

        bkname = row.book_name.strip().lower()

        if bkname not in story_cache:

            if os.path.exists(f"data/story/{bkname}.txt"):
                text = load_story(f"data/story/{bkname}.txt")
                story_cache[bkname] = build_story_index(text)
            else:
                raise FileNotFoundError(bkname)
            
        story_index = story_cache[bkname]
        label, rationale = run_pipeline(story_index, row.content)
        save_result(row.id,label,rationale)

    #print("Accuracy: ",calc_accuracy("data/train.csv","outputs/results.csv"))

    

if __name__ == "__main__":
    main()
