import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

# Carregar os dados
df = pd.read_csv("wine-clustering.csv")

# Exploração inicial
print("Primeiras linhas do dataset:")
print(df.head())

print("\nInformações gerais:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())


df = df.dropna()

# Criar variável alvo a partir do teor alcoólico
# 3 faixas: baixo, médio, alto
y = pd.cut(df["Alcohol"],
           bins=[df["Alcohol"].min()-0.1, 12, 13, df["Alcohol"].max()+0.1],
           labels=["Baixo", "Médio", "Alto"])

#  todas as colunas menos "Alcohol"
X = df.drop(columns=["Alcohol"])

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinamento do modelo
model = DecisionTreeClassifier(random_state=42, max_depth=4)
model.fit(X_train, y_train)

# Visualizar árvore
plt.figure(figsize=(20,10))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Árvore de Decisão - Faixas de Álcool (Vinhos)")
plt.savefig("arvore_wine.png")

# Avaliação
y_pred = model.predict(X_test)

print("\nAcurácia:", accuracy_score(y_test, y_pred))

print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão - Wine")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
