from fastapi.testclient import TestClient
from app.app import app 

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "loaded"
    assert data["prediction_check"] == "passed"