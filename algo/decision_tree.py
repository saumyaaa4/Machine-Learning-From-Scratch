# ================================
# Decision Tree Classifier
# ================================

# Introduction:
# A supervised machine learning algorithm that splits data into branches to make decisions.
# Example: Predicting whether a customer will buy a product based on features.

# ------------------------
# Import Libraries
# ------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
# Replace 'target' with your actual target column
X = data.drop('target', axis=1)  
y = data['target']

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Train Decision Tree
# ------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------------
# Evaluation
# ------------------------
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.2f}")

# ------------------------
# Visualization
# ------------------------
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=True)
plt.title("Decision Tree Visualization")
plt.show()

# ------------------------
# Conclusion / Key Points
# ------------------------
# - Easy to visualize and interpret.
# - Captures non-linear patterns effectively.
# - Prone to overfitting; pruning helps prevent it.
# - Used for classification and regression tasks.
