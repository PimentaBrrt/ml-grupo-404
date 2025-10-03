import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.read_csv("docs/projeto/wine-final.csv", sep=",", encoding="UTF8")

X = df.drop("Wine_Type", axis=1)
X = scaler.fit_transform(X)
y = df["Wine_Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Treino: {X_train.shape[0]} amostras\n")
print(f"Teste: {X_test.shape[0]} amostras\n")
print(f"Proporção: {X_train.shape[0]/X.shape[0]*100:.1f}% treino, {X_test.shape[0]/X.shape[0]*100:.1f}% teste\n")

print("Distribuição das classes - \n")
print("Treino:\n")
print(y_train.value_counts().to_markdown(), "\n")
print("Teste:\n")
print(y_test.value_counts().to_markdown(), "\n")