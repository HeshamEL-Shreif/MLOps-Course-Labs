from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import logging
import joblib
from prometheus_fastapi_instrumentator import Instrumentator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
path = 'svm_model.pkl'
try:
    model = joblib.load(path)
    logging.info(f"Model loaded from {path}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise RuntimeError("Model loading failed")

app = FastAPI(title="Churn_Modeling Classifier API")
logging.info("FastAPI app initialized")
instrumentator = Instrumentator().instrument(app).expose(app)


class InputData(BaseModel):
    features: list[float] 

class PredictionResponse(BaseModel):
    predicted_class: int

@app.get("/")
def root():
    logging.info("Root endpoint accessed")
    return {"message": "Welcome to the Churn_Modeling API!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    expected_features = 13  
    if len(data.features) != expected_features:
        logging.warning(f"Expected {expected_features} features, got {len(data.features)}")
        return {"error": f"Expected {expected_features} features"}
    
    logging.info(f"Prediction request received with features: {data.features}")
    features = np.array(data.features).reshape(1, -1)
    try:
        prediction = model.predict(features)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    return {"predicted_class": int(prediction[0])}



@app.get("/health")
def health_check():
    logging.info("Health check endpoint accessed")

    if model is None:
        logging.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        dummy_input = np.zeros((1, 13)) 
        _ = model.predict(dummy_input)
    except Exception as e:
        logging.error(f"Prediction check failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed")

    return {
        "status": "healthy",
        "model": "loaded",
        "prediction_check": "passed"
    }

@app.get("/home")
def get_model():
    logging.info("Home endpoint accessed")
    return {"model": "svm_model", "version": "1.0", "description": "SVM model for churn prediction",
            "data": "Churn_Modelling.csv", "features": ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"],
            "target": "Exited", "model_type": "SVC", "accuracy": 0.85, "mse": 0.15}


