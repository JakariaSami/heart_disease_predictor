import pandas as pd
import os

df = pd.read_csv("data/raw/heart-disease-cleveland.csv", na_values='?')
df.columns = df.columns.str.strip()
df = df.rename(columns={"diagnosis": "target"})

X = df.drop("target", axis=1)
y = df[["target"]]

os.makedirs("data/raw", exist_ok=True)
X.to_csv("data/raw/features.csv", index=False)
y.to_csv("data/raw/targets.csv", index=False)