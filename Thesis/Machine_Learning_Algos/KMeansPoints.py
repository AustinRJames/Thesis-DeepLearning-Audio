import matplotlib.pyplot as plt
from mlxtend.cluster import Kmeans
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=100, centers=3, random_state=0, cluster_std=3)


plt.scatter(x[:, 0], x[:, 1], s=50)

model = Kmeans(5)
model.fit(x)
y_kmeans = model.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='rainbow')  # Assigns color 'c' based on int y_means

plt.show()