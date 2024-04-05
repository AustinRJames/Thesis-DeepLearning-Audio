import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as cluster_algorithm
from sklearn.cluster import AgglomerativeClustering

shopping_data = pd.read_csv('../CSVs/shopping_data.csv')

data = shopping_data.iloc[: , 3:5].values

plt.figure(figsize=(10,7))
plt.title("Market Segmentation Dendrogram")
dendrogram = cluster_algorithm.dendrogram(cluster_algorithm.linkage(data, method='ward'))

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

plt.figure(figsize=(10,7))
plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap='rainbow')
plt.title("Market Segmentation")
plt.xlabel("Income")
plt.ylabel("Affinity")
plt.show()