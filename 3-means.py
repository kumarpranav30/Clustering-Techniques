import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target


X = data[["petal length (cm)", "petal width (cm)", "sepal width (cm)"]]


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)


predicted_labels = kmeans.labels_


predicted_labels_mapped = predicted_labels.copy()


cluster_to_label = {}
for cluster in range(3):
    true_labels = target[predicted_labels == cluster]
    most_frequent_label = pd.Series(true_labels).mode()[0]
    cluster_to_label[cluster] = most_frequent_label


predicted_labels_mapped = [cluster_to_label[cluster] for cluster in predicted_labels]


accuracy = accuracy_score(target, predicted_labels_mapped)
print(f"Accuracy of k-means clustering on Iris dataset: {accuracy:.2f}")


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


colors = ["blue", "green", "red"]

cluster_map = {0 : "virginica", 1 : "setosa", 2 : "versicolor"}


for cluster in range(3):
    cluster_data = X[predicted_labels == cluster]
    ax.scatter(
        cluster_data.iloc[:, 0],
        cluster_data.iloc[:, 1],
        cluster_data.iloc[:, 2],
        color=colors[cluster],
        label=cluster_map[cluster],
    )


incorrect_indices = target != predicted_labels_mapped
ax.scatter(
    X[incorrect_indices]["petal length (cm)"],
    X[incorrect_indices]["petal width (cm)"],
    X[incorrect_indices]["sepal width (cm)"],
    color="black",
    marker="x",
    label="Incorrectly Classified",
)

ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.set_zlabel("Sepal Width (cm)")
ax.set_title("K-Means Clustering on Iris Dataset (3D)")
ax.legend()
plt.show()
