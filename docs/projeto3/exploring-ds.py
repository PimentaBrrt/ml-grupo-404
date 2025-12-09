import pandas as pd

df = pd.read_csv("docs/projeto3/produtos.csv", sep=";", encoding="UTF8")

print(df.shape)