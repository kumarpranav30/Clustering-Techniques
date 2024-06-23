import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np


iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris.target_names)

selected_features = ['petal length (cm)', 'petal width (cm)', 'sepal width (cm)']
X = data[selected_features]


silhouette_scores = []
db_scores = []
wcss = []


for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    
    
    silhouette = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(silhouette)
    
    
    db = davies_bouldin_score(X, kmeans.labels_)
    db_scores.append(db)
    
    
    wcss.append(kmeans.inertia_)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))


ax1.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', color='b')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Silhouette Score')
ax1.set_title('Silhouette Score for Optimal k')
ax1.grid(True)


ax2.plot(range(2, 11), db_scores, marker='o', linestyle='-', color='r')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Davies-Bouldin Index')
ax2.set_title('Davies-Bouldin Index for Optimal k')
ax2.grid(True)


ax3.plot(range(2, 11), wcss, marker='o', linestyle='-', color='g')
ax3.set_xlabel('Number of Clusters (k)')
ax3.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
ax3.set_title('Elbow Method for Optimal k')
ax3.grid(True)

plt.tight_layout()
plt.show()


best_k_silhouette = np.argmax(silhouette_scores) + 2  
best_k_db = np.argmin(db_scores) + 2 

print(f'Best k based on Silhouette Score: {best_k_silhouette}')
print(f'Best k based on Davies-Bouldin Index: {best_k_db}')


best_k = best_k_silhouette  
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X)
cluster_labels = kmeans.labels_


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X['petal length (cm)'], X['petal width (cm)'], X['sepal width (cm)'],
                     c=cluster_labels, cmap='viridis', edgecolor='k', s=100)

ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.set_zlabel('Sepal Width (cm)')
ax.set_title(f'K-Means Clustering with k={best_k}')
ax.legend(*scatter.legend_elements(), title="Clusters", loc='best')
ax.grid(True)
plt.show()
