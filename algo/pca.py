# ================================
# Principal Component Analysis (PCA)
# ================================

# Introduction:
# Reduces dataset dimensions while keeping most of the information.
# Example: compressing image data or visualizing high-dimensional data.

# ------------------------
# Import Libraries
# ------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ------------------------
# Load Dataset
# ------------------------
# Replace 'dataset.csv' with your dataset file
data = pd.read_csv('dataset.csv')
print("First 5 rows of dataset:")
print(data.head())

# ------------------------
# Data Preparation
# ------------------------
# Select features (X) to reduce dimensions
X = data[['Feature1', 'Feature2', 'Feature3']]  # Replace with actual features

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------
# Apply PCA
# ------------------------
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

# ------------------------
# Visualization
# ------------------------
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: 2D Projection")
plt.show()

# ------------------------
# Conclusion / Key Points
# ------------------------
# - Helps visualize high-dimensional data.
# - Speeds up ML models by reducing features.
# - Retains most important information (variance) while reducing dimensions.
