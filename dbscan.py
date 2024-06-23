import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import random as rd

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target
target_names = iris.target_names


selected_features = ['petal length (cm)', 'petal width (cm)', 'sepal width (cm)']
data_selected = data[selected_features]


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_selected)


eps = 0.354285
min_samples = 3
print(f'eps = {eps}, min_samples = {min_samples}')
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(scaled_data)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']


core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True


plotted_labels = set()

for cluster_label in np.unique(clusters):
    cluster_mask = (clusters == cluster_label)
    if cluster_label == -1:
        ax.scatter(scaled_data[cluster_mask, 0], scaled_data[cluster_mask, 1], scaled_data[cluster_mask, 2], 
                color='black', marker='o', s=50, label='Outliers')
    else:
        ax.scatter(scaled_data[cluster_mask, 0], scaled_data[cluster_mask, 1], scaled_data[cluster_mask, 2], 
                color=colors[cluster_label % len(colors)], marker='o', s=50, label=f'Cluster {cluster_label}')


ax.set_title('DBSCAN Clustering of Iris Dataset')
ax.set_xlabel('Scaled Petal Length (cm)')
ax.set_ylabel('Scaled Petal Width (cm)')
ax.set_zlabel('Scaled Sepal Width (cm)')
ax.legend(loc='best')
plt.show()


