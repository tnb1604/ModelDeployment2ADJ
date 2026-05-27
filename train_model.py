import os
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os.makedirs("model", exist_ok=True)

# v1.0.0 — full model, 100 estimators, no depth limit (deployed)
model_v1 = RandomForestClassifier(n_estimators=100, random_state=42)
model_v1.fit(X_train, y_train)
acc_v1 = accuracy_score(y_test, model_v1.predict(X_test))
print(f"v1.0.0 accuracy: {acc_v1:.2%}")
with open("model/iris_model_v1.0.0.pkl", "wb") as f:
    pickle.dump(model_v1, f)

# v1.1.0 — single shallow tree, tested as a minimal alternative
model_v2 = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=42)
model_v2.fit(X_train, y_train)
acc_v2 = accuracy_score(y_test, model_v2.predict(X_test))
print(f"v1.1.0 accuracy: {acc_v2:.2%}")
with open("model/iris_model_v1.1.0.pkl", "wb") as f:
    pickle.dump(model_v2, f)

print(f"\nDeploying v1.0.0 ({acc_v1:.2%}) — v1.1.0 rejected ({acc_v2:.2%})")
