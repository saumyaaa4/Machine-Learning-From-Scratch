# ==========================================
# K-Means Demo (Using From-Scratch Implementation)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
from algo.k_means import KMeans

# ---------------------------------------
# Load Dataset
# ---------------------------------------
data = pd.read_csv("data/customers.csv")
X = data[['AnnualIncome', 'SpendingScore']].values

# ---------------------------------------
# Train Model
# ---------------------------------------
model = KMeans(k=3, max_iters=100)
model.fit(X)

clusters = model.predict(X)
centers = model.centroids

print("Cluster Centers:")
print(centers)

# ---------------------------------------
# Visualization
# ---------------------------------------
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='X')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")
plt.show()
