import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

encoder = OneHotEncoder()
scaler = StandardScaler()

df = pd.read_csv("docs/projeto3/produtos.csv", sep=";", encoding="UTF8")
droppar = []

features_num = ["preco", "estrelas_media", "avaliacoes"]

for i in range(len(df)):
    if df.loc[i, "recomendacoes"] == 0:
        droppar.append(i)

df = df.drop(index=droppar)
df = df.drop("nome", axis=1)

df["preco"] = df["preco"].str.replace(",", ".").astype(float)
df["estrelas_media"] = df["estrelas_media"].str.replace(",", ".").astype(float)

df_scaled = pd.DataFrame(scaler.fit_transform(df[features_num]), columns=features_num, index=df.index)
df_encoded = encoder.fit_transform(df[["formato"]]).toarray()
df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(["formato"]), index=df.index)

X = pd.concat([df_scaled, df_encoded], axis=1)
y = df["recomendacoes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

melhor_k = None
melhor_r2 = -999
resultados = []

for k in [3, 5, 7, 9, 11, 13, 15]:
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance", p=2)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    resultados.append((k, r2))

    print(f"K = {k} -> R² = {r2:.4f}")

    if r2 > melhor_r2:
        melhor_r2 = r2
        melhor_k = k

print("\nMelhor K encontrado:", melhor_k)
print("Melhor R² obtido:", melhor_r2)

knn_final = KNeighborsRegressor(n_neighbors=melhor_k, weights="distance", p=2)
knn_final.fit(X_train, y_train)

train_accuracy = knn_final.score(X_train, y_train)
test_accuracy = knn_final.score(X_test, y_test)
print(f"\n<b>R² dos conjuntos - KNN</b>\n")
print(f"R² no Treino: {train_accuracy:.4f} \n")
print(f"R² no Teste: {test_accuracy:.4f}")

cv_scores = cross_val_score(knn_final, X, y, cv=5)
print(f"\n<b>Validação Cruzada (5-fold) -</b>\n")
print(f"Scores: {cv_scores}\n")
print(f"Média: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")