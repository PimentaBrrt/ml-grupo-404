import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

scaler = StandardScaler()

df = pd.read_csv("docs/projeto/wine-final.csv", sep=",", encoding="UTF8")

X = df.drop(columns=["Wine_Type", "cluster"], axis=1)
X = scaler.fit_transform(X)
y = df["Wine_Type"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_pca.fit(X_train_pca, y_train)

train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)
print(f"\n<b>Acurácias dos conjuntos -</b>\n")
print(f"Acurácia no Treino: {train_accuracy:.4f} \n")
print(f"Acurácia no Teste: {test_accuracy:.4f}")

cv_scores = cross_val_score(knn, X, y_encoded, cv=5)
print(f"\n<b>Validação Cruzada (5-fold) -</b>\n")
print(f"Scores: {cv_scores}\n")
print(f"Média: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")