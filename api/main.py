from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schema import PatientData, PredictionResponse
from api.predictor import predict

app = FastAPI(
    title="Heart Disease Predictor API",
    description="Predicts heart disease risk from clinical features with SHAP explainability.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Heart Disease Predictor API is running."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(patient: PatientData):
    try:
        result = predict(patient.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))