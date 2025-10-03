import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

scaler = StandardScaler()

df = pd.read_csv("docs/projeto/wine-clustering.csv", sep=",", encoding="UTF8")

X = scaler.fit_transform(df)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X)

kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

silhouette_avg = silhouette_score(X_tsne, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           marker="*", s=300, c="red", label="Centroids")

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title(f"Clusters de Vinhos - t-SNE + K-means (K=3)\nSilhouette: {silhouette_avg:.4f}")
plt.legend()
plt.colorbar(scatter, label="Cluster")

# plt.savefig("docs/projeto/images/k-means-tsne.svg", format="svg", transparent=True)
plt.close()