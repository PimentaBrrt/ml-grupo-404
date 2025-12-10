import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

def calcular_vif(df_num):
    vif = []
    colunas = df_num.columns

    for i in range(len(colunas)):
        y = df_num.iloc[:, i]
        X = df_num.drop(colunas[i], axis=1)

        modelo = LinearRegression().fit(X, y)
        r2 = modelo.score(X, y)
        vif.append(1 / (1 - r2))

    return pd.DataFrame({"Variavel": colunas, "VIF": vif})

encoder = OneHotEncoder()
scaler = StandardScaler()

df = pd.read_csv("docs/projeto3/produtos.csv", sep=";", encoding="UTF8")
droppar = []

for i in range(len(df)):
    if df.loc[i, "recomendacoes"] == 0:
        droppar.append(i)

df = df.drop(index=droppar)
df = df.drop("nome", axis=1)

df["preco"] = df["preco"].str.replace(",", ".").astype(float)
df["estrelas_media"] = df["estrelas_media"].str.replace(",", ".").astype(float)

df_num = df[["preco", "estrelas_media", "avaliacoes"]]

vif_df = calcular_vif(df_num)
print(vif_df.to_markdown())
