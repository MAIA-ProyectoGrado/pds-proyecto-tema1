from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api import model_loader
from src.api.labels import CANONICAL_LABELS, DISPLAY_NAMES, SECTION_OPTIONS, display_name
from src.api.schemas import ModelsResponse, PredictRequest, PredictResponse

ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.warm_up()
    yield


app = FastAPI(
    title="SCIF - API de Inferencia",
    description="Endpoint REST para clasificar la función retórica de citas científicas.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "SCIF Inference API is operational"}


@app.get("/models", response_model=ModelsResponse)
def list_models():
    """Catálogo de modelos servibles. El tablero se puebla desde aquí: no lleva
    ningún modelo ni métrica escritos a mano."""
    return ModelsResponse(
        models=model_loader.catalog(),
        default_model=model_loader.DEFAULT_MODEL,
        labels=CANONICAL_LABELS,
        label_display=DISPLAY_NAMES,
        sections=SECTION_OPTIONS,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_citation(request: PredictRequest):
    if not request.citation_context or not request.citation_context.strip():
        raise HTTPException(status_code=400, detail="El campo citation_context no puede estar vacío.")

    try:
        res = model_loader.predict(
            request.citation_context,
            request.rhetorical_section,
            request.model,
        )
    except model_loader.ModelNotFound:
        disponibles = ", ".join(model_loader.REGISTRY)
        raise HTTPException(
            status_code=422,
            detail=f"Modelo '{request.model}' desconocido. Disponibles: {disponibles}.",
        )
    except model_loader.ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {exc}")

    return PredictResponse(
        predicted_label=res["predicted_label"],
        predicted_label_display=display_name(res["predicted_label"]),
        confidence=res["confidence"],
        probabilities=res["probabilities"],
        ambiguous=res["ambiguous"],
        model_id=res["model_id"],
        input_text=res["input_text"],
        latency_ms=res["latency_ms"],
        status="success",
    )
