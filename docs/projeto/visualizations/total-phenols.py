import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/projeto/wine-clustering.csv", sep=",", encoding="UTF8")

plt.figure(figsize=(10, 6))
plt.hist(df["Total_Phenols"], bins=30, edgecolor="black", alpha=0.7, color="purple")
plt.title("Distribuição do Total de Fenóis dos Vinhos - Histograma")
plt.xlabel("Total de Fenóis")
plt.ylabel("Frequência")
plt.grid(axis="y", alpha=0.3)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()