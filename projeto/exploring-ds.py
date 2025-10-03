import pandas as pd

df = pd.read_csv("docs/projeto/wine-clustering.csv", sep=",", encoding="UTF8")

print(df.shape)