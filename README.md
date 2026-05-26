# Iris Classifier API — Team A2

A REST API that serves an Iris flower classification model. Built with Flask and deployed on Azure Container Apps as part of the Model Deployment course at Inholland HBO IT.

## What it does

Send measurements of an Iris flower (sepal/petal length and width) and get back the predicted species along with confidence scores for each class.

## Live deployment

```
https://team-a2-iris-api.ambitiouswater-7f7ad293.westeurope.azurecontainerapps.io
```

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Returns whether the API is up and the model is loaded |
| GET | /metadata | Model info, feature names, and valid input ranges |
| POST | /predict | Make a prediction |
| GET | /metrics | Prometheus metrics |

### Example predict request

```bash
curl -X POST https://team-a2-iris-api.ambitiouswater-7f7ad293.westeurope.azurecontainerapps.io/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Response:
```json
{
  "prediction": "setosa",
  "prediction_index": 0,
  "probabilities": { "setosa": 1.0, "versicolor": 0.0, "virginica": 0.0 },
  "model_version": "1.0.0"
}
```

## Running locally

```bash
pip install -r requirements.txt
python train_model.py       # generates model/iris_model_v1.0.0.pkl
python app.py               # starts Flask dev server on port 5000
```

Or with Docker:

```bash
docker build -t iris-api .
docker run -p 5000:5000 iris-api
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_PATH | model/iris_model_v1.0.0.pkl | Path to the model file |
| MODEL_VERSION | 1.0.0 | Version string returned in API responses |
| FLASK_DEBUG | false | Enable Flask debug mode |

## Running tests

```bash
pytest tests/ -v
```

## Model versioning

Model files follow semantic versioning: `iris_model_v<major>.<minor>.<patch>.pkl`. The current version is `v1.0.0`. When the model is retrained, the file is renamed with the new version and `MODEL_VERSION` is updated accordingly.

## CI/CD

On every push to `main`, GitHub Actions:
1. Runs pytest — build is blocked if any test fails
2. Builds a Docker image and pushes it to Azure Container Registry with both a `latest` tag and a git SHA tag for traceability
