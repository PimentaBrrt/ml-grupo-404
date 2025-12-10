import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

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

print(X.shape)
print(X.head())