from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

path = './svm_model.pkl'
path = '/Users/heshamelsherif/Docs/iti/MLOps/MLOps-Course-Labs/app/svm_model.pkl'
import joblib

model = joblib.load(path)
# Load the model
# with open(path, "rb") as f:
#     model = pickle.load(f)
app = FastAPI(title="Churn_Modeling Classifier API")


class InputData(BaseModel):
    features: list[float] 

class PredictionResponse(BaseModel):
    predicted_class: int

@app.get("/")
def root():
    return {"message": "Welcome to the Churn_Modeling API!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"predicted_class": int(prediction[0])}


