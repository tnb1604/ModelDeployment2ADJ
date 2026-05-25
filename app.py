"""
app.py
------
Flask API for the Iris flower classifier.
Endpoints:
    GET  /health    - Health check
    GET  /metadata  - Model metadata
    POST /predict   - Make a prediction
"""

import logging
import pickle
import os
from flask import Flask, jsonify, request

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "model/iris_model.pkl")
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Load model at startup so we fail fast if it's missing
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", MODEL_PATH)
except FileNotFoundError:
    logger.error("Model file not found at %s — run train_model.py first", MODEL_PATH)
    model = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — returns 200 if the model is loaded and ready."""
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "model not loaded"}), 503
    return jsonify({"status": "healthy"}), 200


@app.route("/metadata", methods=["GET"])
def metadata():
    """Returns information about the model and expected inputs."""
    return jsonify({
        "model": "RandomForestClassifier",
        "dataset": "Iris",
        "version": "1.0.0",
        "classes": CLASS_NAMES,
        "features": FEATURE_NAMES,
        "feature_units": "cm",
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with the four Iris features and returns the predicted class.

    Example request body:
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # Validate all required features are present
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return jsonify({
            "error": f"Missing required fields: {missing}",
            "required_fields": FEATURE_NAMES,
        }), 400

    # Validate all values are numbers
    try:
        features = [[float(data[f]) for f in FEATURE_NAMES]]
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numbers"}), 400

    # Predict
    try:
        prediction_index = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist()
        predicted_class = CLASS_NAMES[prediction_index]

        logger.info("Prediction: %s (index %d)", predicted_class, prediction_index)

        return jsonify({
            "prediction": predicted_class,
            "prediction_index": prediction_index,
            "probabilities": {
                CLASS_NAMES[i]: round(p, 4)
                for i, p in enumerate(probabilities)
            },
        }), 200

    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        return jsonify({"error": "Prediction failed. Please check your input."}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
