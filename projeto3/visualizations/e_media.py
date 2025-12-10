import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

df = pd.read_csv("docs/projeto3/produtos.csv", sep=";", encoding="UTF8")

df["estrelas_media"] = df["estrelas_media"].str.replace(",", ".").astype(float)

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="estrelas_media", y="recomendacoes", color="orange")
plt.title("Distribuição entre Média de Estrelas e Recomendações")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()