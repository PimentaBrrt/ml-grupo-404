import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

df = pd.read_csv("docs/projeto3/produtos.csv", sep=";", encoding="UTF8")

sns.histplot(data=df, x="recomendacoes", bins=20)
plt.title("Histograma de Recomendações")
plt.xlabel("Recomendações")
plt.ylabel("Frequência")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()

