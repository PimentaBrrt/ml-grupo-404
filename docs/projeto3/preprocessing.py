import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
scaler = StandardScaler()

df = pd.read_csv("docs/projeto3/produtos.csv", sep=";", encoding="UTF8")
droppar = []

features_num = ["preco", "estrelas_media", "avaliacoes", "recomendacoes"]
categorical_cols = ["formato", "aminoacidos", "carboidratos", "clinical", "proteinas", "termogenicos", "veganos", "vegetarianos", "vitaminas"]

for i in range(len(df)):
    if df.loc[i, "recomendacoes"] == 0:
        droppar.append(i)

df = df.drop(index=droppar)
df = df.drop("nome", axis=1)

df["preco"] = df["preco"].str.replace(",", ".").astype(float)
df["estrelas_media"] = df["estrelas_media"].str.replace(",", ".").astype(float)

df_scaled = scaler.fit_transform(df[features_num])
df_encoded = encoder.fit_transform(df[categorical_cols])

df_final = pd.concat([pd.DataFrame(df_scaled), pd.DataFrame(df_encoded)], axis=1)

print(df_final.shape)
print(df_final.head())