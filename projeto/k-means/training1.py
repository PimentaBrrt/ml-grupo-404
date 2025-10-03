import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

scaler = StandardScaler()

df = pd.read_csv("docs/projeto/wine-clustering.csv", sep=",", encoding="UTF8")

X = scaler.fit_transform(df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

silhouette_avg = silhouette_score(X, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="*", s=300, c="red", label="Centroids")

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var.)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var.)")
plt.title("Clusters de Vinhos - K-means (K=3)")
plt.legend()
plt.colorbar(scatter, label="Cluster")

# plt.savefig("docs/projeto/images/k-means.svg", format="svg", transparent=True)
plt.close()