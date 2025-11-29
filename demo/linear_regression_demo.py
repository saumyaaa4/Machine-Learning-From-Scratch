# ==========================================
# Linear Regression Demo (Using From-Scratch Implementation)
# ==========================================

import pandas as pd
from algo.linear_regression import LinearRegression
import matplotlib.pyplot as plt

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv("data/housing.csv")
X = data[['Area', 'Bedrooms', 'Age']]
y = data['Price']

# -------------------------------
# Train-test split
# -------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train model
# -------------------------------
model = LinearRegression(learning_rate=0.00000001, epochs=500)  # tune if needed
model.fit(X_train.values, y_train.values)

# -------------------------------
# Predictions
# -------------------------------
predictions = model.predict(X_test.values)

print("\nActual Prices:")
print(list(y_test.values))

print("\nPredicted Prices:")
print(list(predictions))

# -------------------------------
# Optional Visualization
# -------------------------------
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression Predictions vs Actual")
plt.show()

