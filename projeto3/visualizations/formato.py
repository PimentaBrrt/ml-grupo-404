import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

df = pd.read_csv("docs/projeto3/produtos.csv", sep=";", encoding="UTF8")

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="formato", color="salmon")

plt.title("Frequência de Produtos por Formato")
plt.xlabel("Formato")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()