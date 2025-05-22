from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Churn_Modeling API!"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == "loaded"
    assert response.json()["prediction_check"] == "passed"

def test_get_model():
    response = client.get("/home")
    assert response.status_code == 200
    assert response.json()["model"] == "svm_model"
    assert "features" in response.json()

def test_valid_prediction():
    test_input = {
        "features": [600.0, 1, 0, 40.0, 3, 60000.0, 2, 1, 1, 50000.0, 1, 0, 0]
    }
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    assert "predicted_class" in response.json()
    assert isinstance(response.json()["predicted_class"], int)


def test_invalid_prediction_too_few_features():
    test_input = {
        "features": [600.0, 1, 0] 
    }
    response = client.post("/predict", json=test_input)
    assert response.status_code == 400
    assert "error" in response.json()