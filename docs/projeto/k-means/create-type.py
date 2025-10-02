import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

scaler = StandardScaler()

df = pd.read_csv("docs/projeto/wine-clustering.csv", sep=",", encoding="UTF8")

X = scaler.fit_transform(df)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X)

kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

df["cluster"] = cluster_labels
df["Wine_Type"] = [f"Wine Type {label + 1}" for label in cluster_labels]

print("Distribuição dos clusters -<br>")
for i in range(1, 4):
    count = len(df[df['Wine_Type'] == f'Wine Type {i}'])
    print(f"Wine Type {i}: {count}<br>")
    
# Salvar para csv
# df.to_csv("wine-final.csv", index=False)