import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

df = pd.read_csv("docs/projeto3/produtos.csv", sep=";", encoding="UTF8")

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="avaliacoes", y="recomendacoes", color="lightgreen")
plt.title("Distribuição entre Total de Avaliações e Recomendações")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()