import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

df = pd.read_csv("docs/projeto3/produtos.csv", sep=";", encoding="UTF8")

categorias = ["aminoacidos", "carboidratos", "clinical", "proteinas", "termogenicos", "veganos", "vegetarianos", "vitaminas"]

freq = df[categorias].sum().reset_index()
freq.columns = ["categoria", "frequencia"]

plt.figure(figsize=(10, 6))
sns.barplot(data=freq, x="categoria", y="frequencia", color="orange")

plt.title("Frequência de Presença das Categorias nos Produtos")
plt.xlabel("Categoria")
plt.ylabel("Número de Produtos")
plt.xticks(rotation=45)
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()