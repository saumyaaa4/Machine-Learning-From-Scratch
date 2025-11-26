# ================================
# Decision Tree
# ================================

# Introduction:
# Splits data into branches to make decisions.
# Example: classify animals based on features like weight, legs, and type.

# ------------------------
# Import Libraries
# ------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
# Select features (X) and target (y)
X = data[['Feature1', 'Feature2']]  # Replace with actual feature columns
y = data['Target']                  # Replace with target column

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# Train Model
# ------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------------
# Make Predictions
# ------------------------
y_pred = model.predict(X_test)

# ------------------------
# Evaluate Model
# ------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------
# Visualization
# ------------------------
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=X.columns, class_names=True, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# ------------------------
# Conclusion / Key Points
# ------------------------
# - Easy to interpret and visualize.
# - Can handle both categorical and numerical data.
# - May overfit if tree is too deep; use pruning or ensemble methods.
