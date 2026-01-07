# src/store.py
import pathway as pw
import pandas as pd

def chunks_to_table(chunks):
    """
    Converts chunk list into a Pathway table.
    """
    df = pd.DataFrame(chunks)
    table = pw.Table.from_pandas(df)
    return table
