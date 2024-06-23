import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Perform PCA
pca = PCA(n_components=4)  # Reduce to 2 principal components
principal_components = pca.fit_transform(data)

# Variance explained by each principal component
explained_variance = pca.explained_variance_ratio_

# Plot the scree plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.8, align='center', label='Explained Variance')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='Cumulative Explained Variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Print contribution of each original feature to each principal component
components_df = pd.DataFrame(pca.components_, columns=iris.feature_names)
print("\nContribution of each original feature to each principal component:")
print(components_df)

# Plot the top 2 Principal Components
plt.figure(figsize=(10, 8))
targets = iris.target_names
colors = ['r', 'g', 'b']

for target, color in zip(targets, colors):
    indices_to_keep = iris.target == targets.tolist().index(target)
    plt.scatter(principal_components[indices_to_keep, 0], principal_components[indices_to_keep, 1], c=color, s=50)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset - Top 2 PCs')
plt.legend(targets)
plt.grid(True)
plt.show()
