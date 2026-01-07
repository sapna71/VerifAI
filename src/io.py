# src/io.py

def load_story(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_backstory(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_result(story_id, label, rationale, out_path="results.csv"):
    import csv
    header = ["story_id", "prediction", "rationale"]
    row = [story_id, label, rationale]

    try:
        with open(out_path, "x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)
    except FileExistsError:
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
