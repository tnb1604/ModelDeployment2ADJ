"""
app.py
------
Flask API for the Iris flower classifier.
Endpoints:
    GET  /health    - Health check
    GET  /metadata  - Model metadata
    POST /predict   - Make a prediction
    GET  /metrics   - Prometheus metrics (operational + functional)
"""

import logging
import pickle
import os
import time
from flask import Flask, jsonify, request, g
from flask_cors import CORS
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from any frontend

MODEL_PATH = os.getenv("MODEL_PATH", "model/iris_model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Realistic value ranges for Iris features (cm), used for input validation
FEATURE_RANGES = {
    "sepal_length": (4.0, 8.0),
    "sepal_width":  (2.0, 5.0),
    "petal_length": (1.0, 7.0),
    "petal_width":  (0.1, 3.0),
}

# ── Prometheus metrics ────────────────────────────────────────────────────────
# Operational metrics (request count, latency, status codes) are tracked
# automatically by PrometheusMetrics on the /metrics endpoint.
metrics = PrometheusMetrics(app)
metrics.info("iris_api_info", "Iris classifier API", version=MODEL_VERSION)

# Functional metrics — track model behaviour, not just HTTP traffic
prediction_counter = Counter(
    "iris_predictions_total",
    "Total number of predictions made, labelled by predicted class",
    ["predicted_class"],
)
confidence_histogram = Histogram(
    "iris_prediction_confidence",
    "Confidence (probability) of the predicted class",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
invalid_input_counter = Counter(
    "iris_invalid_inputs_total",
    "Total number of requests rejected due to invalid or out-of-range input",
    ["reason"],
)

# ── Model loading ─────────────────────────────────────────────────────────────
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s (version %s)", MODEL_PATH, MODEL_VERSION)
except FileNotFoundError:
    logger.error("Model file not found at %s — run train_model.py first", MODEL_PATH)
    model = None


# ── Request lifecycle hooks ───────────────────────────────────────────────────

@app.before_request
def start_timer():
    """Record request start time for latency logging."""
    g.start_time = time.time()


@app.after_request
def log_request(response):
    """Log every request with method, path, status code, and duration."""
    duration_ms = round((time.time() - g.start_time) * 1000, 2)
    logger.info(
        "%s %s → %d (%.2f ms)",
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )
    return response


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "available_endpoints": [
        "GET /health", "GET /metadata", "POST /predict", "GET /metrics"
    ]}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — returns 200 if the model is loaded and ready."""
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "model not loaded"}), 503
    return jsonify({"status": "healthy", "model_version": MODEL_VERSION}), 200


@app.route("/metadata", methods=["GET"])
def metadata():
    """Returns information about the model and expected inputs."""
    return jsonify({
        "model": "RandomForestClassifier",
        "dataset": "Iris",
        "version": MODEL_VERSION,
        "classes": CLASS_NAMES,
        "features": FEATURE_NAMES,
        "feature_units": "cm",
        "feature_ranges": FEATURE_RANGES,
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
        invalid_input_counter.labels(reason="invalid_json").inc()
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # Validate all required features are present
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        invalid_input_counter.labels(reason="missing_fields").inc()
        return jsonify({
            "error": f"Missing required fields: {missing}",
            "required_fields": FEATURE_NAMES,
        }), 400

    # Validate all values are numbers
    try:
        features = [[float(data[f]) for f in FEATURE_NAMES]]
    except (TypeError, ValueError):
        invalid_input_counter.labels(reason="non_numeric").inc()
        return jsonify({"error": "All feature values must be numbers"}), 400

    # Validate values are within realistic Iris measurement ranges
    out_of_range = []
    for feature in FEATURE_NAMES:
        val = float(data[feature])
        low, high = FEATURE_RANGES[feature]
        if not (low <= val <= high):
            out_of_range.append(
                f"{feature}={val} (expected {low}–{high} cm)"
            )
    if out_of_range:
        invalid_input_counter.labels(reason="out_of_range").inc()
        logger.warning("Out-of-range input values: %s", out_of_range)
        return jsonify({
            "error": "One or more feature values are outside the expected range",
            "out_of_range": out_of_range,
            "feature_ranges": FEATURE_RANGES,
        }), 422

    # Predict
    try:
        prediction_index = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist()
        predicted_class = CLASS_NAMES[prediction_index]
        confidence = probabilities[prediction_index]

        # Track functional metrics
        prediction_counter.labels(predicted_class=predicted_class).inc()
        confidence_histogram.observe(confidence)

        logger.info(
            "Prediction: %s (confidence %.2f) | input: %s",
            predicted_class,
            confidence,
            {f: data[f] for f in FEATURE_NAMES},
        )

        return jsonify({
            "prediction": predicted_class,
            "prediction_index": prediction_index,
            "probabilities": {
                CLASS_NAMES[i]: round(p, 4)
                for i, p in enumerate(probabilities)
            },
            "model_version": MODEL_VERSION,
        }), 200

    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        return jsonify({"error": "Prediction failed. Please check your input."}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=5000)
