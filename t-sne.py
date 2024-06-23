import pandas as pd
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target
target_names = iris.target_names

# Apply t-SNE to reduce the dimensionality to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(data)

# Plotting the t-SNE visualization with color-coded species
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red']
for i, target_name in enumerate(target_names):
    plt.scatter(X_tsne[target == i, 0], X_tsne[target == i, 1], color=colors[i], label=target_name)

plt.title('t-SNE Visualization of Iris Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.grid(True)
plt.show()
