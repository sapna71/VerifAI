import pandas as pd

def load_dataset(path):
    df=pd.read_csv(path)
    return df

def load_story(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_backstory(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_result(story_id, label, rationale, out_path="outputs/results.csv"):
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

def calc_accuracy(path_train,path_res):
    df1,df2=load_dataset(path_train),load_dataset(path_res)
    df_merged = df1.merge(df2, left_on="id", right_on="story_id")
    label={"consistent":1,"contradict":0}
    accuracy=0
    for idx,row in df_merged.iterrows():
        if label[row["label"].strip().lower()]==row["prediction"]:
            accuracy+=1
    return accuracy/len(df_merged)
        
