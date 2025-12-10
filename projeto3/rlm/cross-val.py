import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score

def r2_adj(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def stepwise_backward(X, y):
    selecionadas = list(X.columns)
    n = len(y)
    modelo_inicial = LinearRegression().fit(X, y)
    melhor_r2_adj = r2_adj(modelo_inicial.score(X, y), n, len(selecionadas))

    melhorou = True

    while melhorou and len(selecionadas) > 1:
        r2_adj_temp = []
        for var in selecionadas:
            teste_vars_sel = [v for v in selecionadas if v != var]
            modelo = LinearRegression().fit(X[teste_vars_sel], y)
            r2 = modelo.score(X[teste_vars_sel], y)
            r2_adj_temp.append((var, r2_adj(r2, n, len(teste_vars_sel))))

        pior_var, melhor_r2 = max(r2_adj_temp, key=lambda x: x[1])

        if melhor_r2 > melhor_r2_adj:
            melhor_r2_adj = melhor_r2
            selecionadas.remove(pior_var)
        else:
            melhorou = False

    return selecionadas

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

vars_sel = stepwise_backward(X_train, y_train)

rlm = LinearRegression()
rlm.fit(X_train[vars_sel], y_train)

train_accuracy = rlm.score(X_train[vars_sel], y_train)
test_accuracy = rlm.score(X_test[vars_sel], y_test)
print(f"\n<b>R² dos conjuntos - Regressão Linear Múltipla</b>\n")
print(f"R² no Treino: {train_accuracy:.4f} \n")
print(f"R² no Teste: {test_accuracy:.4f}")

cv_scores = cross_val_score(rlm, X, y, cv=5)
print(f"\n<b>Validação Cruzada (5-fold) -</b>\n")
print(f"Scores: {cv_scores}\n")
print(f"Média: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")