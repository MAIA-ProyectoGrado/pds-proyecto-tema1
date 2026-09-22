import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api import model_loader, retrieval
from src.api.labels import CANONICAL_LABELS, DISPLAY_NAMES, SECTION_OPTIONS, display_name
from src.api.schemas import (
    ModelsResponse,
    PredictRequest,
    PredictResponse,
    RetrieveRequest,
    RetrieveResponse,
)

# Orígenes permitidos. En el despliegue con Docker el navegador solo habla con
# nginx (mismo origen), así que CORS no interviene; la variable sirve para los
# entornos donde el tablero y la API se sirven por separado.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "SCIF_ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080"
    ).split(",")
    if o.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.warm_up()
    yield


app = FastAPI(
    title="SCIF - API de Inferencia",
    description=(
        "Clasificación de la función retórica de citas científicas y recuperación "
        "local de los pasajes más relevantes del artículo citado."
    ),
    version="3.0.0",
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
        retrieval_available=retrieval.retrieval_available(),
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


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_passages(request: RetrieveRequest):
    """Top-k pasajes del artículo citado más similares al contexto de cita.
    El artículo se indica por id/URL de arXiv o pegando su texto completo."""
    if not request.citation_context or not request.citation_context.strip():
        raise HTTPException(status_code=400, detail="El campo citation_context no puede estar vacío.")
    has_id = bool(request.arxiv_id and request.arxiv_id.strip())
    has_text = bool(request.cited_text and request.cited_text.strip())
    if not has_id and not has_text:
        raise HTTPException(
            status_code=400,
            detail="Indica el artículo citado: arxiv_id (id o URL de arXiv) o cited_text (texto completo).",
        )

    try:
        res = retrieval.retrieve(
            request.citation_context,
            arxiv_id=request.arxiv_id if has_id else None,
            cited_text=request.cited_text if has_text else None,
            top_k=request.top_k,
        )
    except retrieval.InvalidArxivId as exc:
        raise HTTPException(status_code=422, detail=f"'{exc}' no es un identificador de arXiv válido.")
    except retrieval.PaperNotFound as exc:
        raise HTTPException(status_code=404, detail=f"arXiv no tiene ningún artículo con id '{exc}'.")
    except retrieval.TextTooShort as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except retrieval.DownloadError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except model_loader.ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"La recuperación requiere SciBERT. {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error durante la recuperación: {exc}")

    return RetrieveResponse(**res, status="success")
