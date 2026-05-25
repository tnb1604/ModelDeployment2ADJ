"""
train_model.py
--------------
Trains a simple Iris flower classifier and saves it to model/iris_model.pkl.
Run this once locally before building your Docker image:
    python train_model.py
"""

import os
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model accuracy: {accuracy:.2%}")

# Save the model
os.makedirs("model", exist_ok=True)
model_path = "model/iris_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")
