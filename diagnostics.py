import pandas as pd

df = pd.read_csv("data/train.csv")

for col in df.columns:
    print(f"{col:30} {df[col].dtype}")