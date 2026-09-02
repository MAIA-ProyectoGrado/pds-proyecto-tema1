from fastapi import FastAPI, HTTPException
from api.schemas import PredictRequest, PredictResponse
from api.model_loader import predictor

app = FastAPI(
    title="SCIF - API de Inferencia",
    description="Endpoint REST para clasificar la función retórica de citas científicas.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok", "message": "SCIF Inference API is operational"}

@app.post("/predict", response_model=PredictResponse)
def predict_citation(request: PredictRequest):
    if not request.citation_context or not request.citation_context.strip():
        raise HTTPException(status_code=400, detail="El campo citation_context no puede estar vacío.")
    
    try:
        res = predictor.predict(request.citation_context)
        return PredictResponse(
            predicted_label=res["predicted_label"],
            confidence=res["confidence"],
            probabilities=res["probabilities"],
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {str(e)}")