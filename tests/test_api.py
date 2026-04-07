from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_PATIENT = {
    "age": 55, "sex": 1, "cp": 3, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 2, "ca": 1, "thal": 2
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_returns_200():
    response = client.post("/predict", json=SAMPLE_PATIENT)
    assert response.status_code == 200

def test_predict_response_structure():
    response = client.post("/predict", json=SAMPLE_PATIENT)
    data = response.json()
    assert "prediction" in data
    assert "prediction_label" in data
    assert "probability" in data
    assert "shap_values" in data

def test_predict_label_matches_prediction():
    response = client.post("/predict", json=SAMPLE_PATIENT)
    data = response.json()
    if data["prediction"] == 1:
        assert data["prediction_label"] == "Disease"
    else:
        assert data["prediction_label"] == "No Disease"

def test_invalid_input_returns_422():
    response = client.post("/predict", json={"age": 999})
    assert response.status_code == 422