import pytest
import app as app_module
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# /health

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_healthy_status(client):
    data = client.get("/health").get_json()
    assert data["status"] == "healthy"


# /metadata

def test_metadata_returns_200(client):
    response = client.get("/metadata")
    assert response.status_code == 200


def test_metadata_contains_required_fields(client):
    data = client.get("/metadata").get_json()
    assert "model" in data
    assert "classes" in data
    assert "features" in data
    assert "version" in data
    assert "feature_ranges" in data


def test_metadata_classes_are_correct(client):
    data = client.get("/metadata").get_json()
    assert data["classes"] == ["setosa", "versicolor", "virginica"]


def test_metadata_features_are_correct(client):
    data = client.get("/metadata").get_json()
    assert data["features"] == [
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    ]


# /predict - valid inputs

VALID_SETOSA = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

VALID_VIRGINICA = {
    "sepal_length": 6.7,
    "sepal_width": 3.0,
    "petal_length": 5.2,
    "petal_width": 2.3,
}


def test_predict_returns_200_for_valid_input(client):
    response = client.post("/predict", json=VALID_SETOSA)
    assert response.status_code == 200


def test_predict_returns_prediction_field(client):
    data = client.post("/predict", json=VALID_SETOSA).get_json()
    assert "prediction" in data
    assert data["prediction"] in ["setosa", "versicolor", "virginica"]


def test_predict_returns_probabilities(client):
    data = client.post("/predict", json=VALID_SETOSA).get_json()
    assert "probabilities" in data
    probs = data["probabilities"]
    assert set(probs.keys()) == {"setosa", "versicolor", "virginica"}
    assert abs(sum(probs.values()) - 1.0) < 0.001


def test_predict_returns_model_version(client):
    data = client.post("/predict", json=VALID_SETOSA).get_json()
    assert "model_version" in data


def test_predict_setosa_correctly_classified(client):
    data = client.post("/predict", json=VALID_SETOSA).get_json()
    assert data["prediction"] == "setosa"


def test_predict_virginica_correctly_classified(client):
    data = client.post("/predict", json=VALID_VIRGINICA).get_json()
    assert data["prediction"] == "virginica"


# /predict - error cases

def test_predict_returns_400_for_empty_body(client):
    response = client.post("/predict", data="", content_type="application/json")
    assert response.status_code == 400


def test_predict_returns_400_for_missing_fields(client):
    response = client.post("/predict", json={"sepal_length": 5.1})
    assert response.status_code == 400
    data = response.get_json()
    assert "missing" in data["error"].lower() or "required" in data["error"].lower()


def test_predict_returns_400_for_non_numeric_values(client):
    response = client.post("/predict", json={
        "sepal_length": "big",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    })
    assert response.status_code == 400


def test_predict_returns_422_for_out_of_range_values(client):
    response = client.post("/predict", json={
        "sepal_length": 99.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    })
    assert response.status_code == 422


def test_predict_422_response_includes_out_of_range_details(client):
    data = client.post("/predict", json={
        "sepal_length": 99.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }).get_json()
    assert "out_of_range" in data


# 404 / 405

def test_unknown_endpoint_returns_404(client):
    response = client.get("/nonexistent")
    assert response.status_code == 404


def test_get_on_predict_returns_405(client):
    response = client.get("/predict")
    assert response.status_code == 405


# API key authentication

def test_predict_returns_401_when_api_key_set_and_missing(client, monkeypatch):
    monkeypatch.setattr(app_module, "API_KEY", "secret123")
    response = client.post("/predict", json=VALID_SETOSA)
    assert response.status_code == 401


def test_predict_returns_401_when_api_key_wrong(client, monkeypatch):
    monkeypatch.setattr(app_module, "API_KEY", "secret123")
    response = client.post("/predict", json=VALID_SETOSA, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_predict_succeeds_with_correct_api_key(client, monkeypatch):
    monkeypatch.setattr(app_module, "API_KEY", "secret123")
    response = client.post("/predict", json=VALID_SETOSA, headers={"X-API-Key": "secret123"})
    assert response.status_code == 200


def test_health_is_public_even_when_api_key_set(client, monkeypatch):
    monkeypatch.setattr(app_module, "API_KEY", "secret123")
    response = client.get("/health")
    assert response.status_code == 200
