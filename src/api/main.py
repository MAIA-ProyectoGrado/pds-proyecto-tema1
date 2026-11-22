import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import config, key_sentences, model_loader, retrieval
from src.api.labels import CANONICAL_LABELS, DISPLAY_NAMES, SECTION_OPTIONS, display_name
from src.api.limits import RateLimiter
from src.api.schemas import (
    HealthResponse,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
    RetrieveRequest,
    RetrieveResponse,
)

API_VERSION = "4.0.0"
STARTED = time.monotonic()
#: rutas que cuentan para el límite de peticiones (las que hacen trabajo pesado)
LIMITED_PATHS = {"/predict", "/retrieve"}
limiter = RateLimiter(config.RATE_LIMIT_PER_MIN)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.warm_up()
    yield


app = FastAPI(
    title="SCIF - API de Inferencia",
    description=(
        "Clasificación de la función retórica de citas científicas y recuperación "
        "local de los pasajes más relevantes del artículo citado.\n\n"
        "`POST /predict` clasifica una cita. Con SciBERT, si se indica el artículo "
        "citado (`arxiv_id` o `cited_text`), el modelo lee además 3 oraciones clave de "
        "su abstract. `POST /retrieve` devuelve los pasajes del artículo citado más "
        "parecidos al contexto. `GET /health` informa el estado del servicio."
    ),
    version=API_VERSION,
    root_path=config.ROOT_PATH,
    lifespan=lifespan,
)

# ── uso controlado ───────────────────────────────────────────────────────────

@app.middleware("http")
async def controlled_use(request: Request, call_next):
    """Tamaño máximo de la petición (413) y límite de peticiones por IP (429)."""
    if request.method == "POST":
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > config.MAX_BODY_KB * 1024:
            return JSONResponse(
                status_code=413,
                content={"detail": f"La petición supera el máximo de {config.MAX_BODY_KB} KB."},
            )
        if request.url.path in LIMITED_PATHS:
            ip = request.client.host if request.client else "desconocida"
            wait = limiter.hit(ip)
            if wait:
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(wait)},
                    content={"detail": (
                        f"Demasiadas peticiones: el máximo es {limiter.per_minute} por minuto. "
                        f"Intenta de nuevo en {wait} s.")},
                )
    return await call_next(request)


# CORS se registra después para que envuelva también las respuestas 413 y 429.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def readable_validation_error(request: Request, exc: RequestValidationError):
    """422 con un mensaje legible, en el mismo formato que los demás errores."""
    messages = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err.get("loc", []) if p != "body") or "petición"
        if err.get("type") == "string_too_long":
            limit = (err.get("ctx") or {}).get("max_length")
            messages.append(f"El campo {field} admite como máximo {limit} caracteres.")
        elif err.get("type") == "missing":
            messages.append(f"Falta el campo {field}.")
        else:
            messages.append(f"{field}: {err.get('msg')}")
    return JSONResponse(status_code=422, content={"detail": " ".join(messages)})


# ── estado ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "SCIF Inference API is operational"}


@app.get("/health", response_model=HealthResponse)
def health():
    """Estado del servicio: qué modelos están en disco y cargados, y la
    configuración efectiva. Responde 200 aunque falte algo (status=degraded)."""
    models = {
        mid: {"available": model_loader.is_available(mid), "loaded": mid in model_loader._cache}
        for mid in model_loader.REGISTRY
    }
    default = model_loader.DEFAULT_MODEL
    ok = models[default]["available"] and (
        not model_loader.REGISTRY[default]["reads_passages"] or model_loader.encoder_available()
    )
    return HealthResponse(
        status="ok" if ok else "degraded",
        version=API_VERSION,
        uptime_s=round(time.monotonic() - STARTED, 1),
        default_model=default,
        models=models,
        key_sentences_encoder={
            "available": model_loader.encoder_available(),
            "loaded": model_loader.encoder_loaded(),
        },
        retrieval_available=retrieval.retrieval_available(),
        config=config.summary(),
    )


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
        key_sentences_available=model_loader.encoder_available(),
    )


# ── clasificación ────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict_citation(request: PredictRequest):
    """Clasifica la función de una cita. El artículo citado es opcional: si se
    indica y el modelo lo lee (SciBERT), se usan 3 oraciones clave de su abstract.
    Si no se puede obtener, se clasifica sin ellas y `key_sentences.detail` lo explica."""
    context = (request.citation_context or "").strip()
    if not context:
        raise HTTPException(status_code=400, detail="El campo citation_context no puede estar vacío.")
    if request.model not in model_loader.REGISTRY:
        disponibles = ", ".join(model_loader.REGISTRY)
        raise HTTPException(
            status_code=422,
            detail=f"Modelo '{request.model}' desconocido. Disponibles: {disponibles}.",
        )
    if not model_loader.is_available(request.model):
        raise HTTPException(
            status_code=503,
            detail=f"No hay pesos para '{request.model}' en {model_loader.model_dir(request.model)}. "
                   "Ejecuta `dvc pull` o define SCIF_MODELS_DIR.",
        )
    section = (request.rhetorical_section or "").strip() or None
    meta = model_loader.REGISTRY[request.model]

    started = time.perf_counter()
    try:
        ks = key_sentences.build(
            context,
            reads_passages=meta["reads_passages"],
            arxiv_id=request.arxiv_id,
            cited_text=request.cited_text,
        )
        res = model_loader.predict(context, section, request.model, ks.texts, ks.title)
    except model_loader.ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {exc}")
    total_ms = round((time.perf_counter() - started) * 1000, 1)

    x = res["model_input"]
    input_text = f"{x[0]} [SEP] {x[1]}" if isinstance(x, tuple) else x
    return PredictResponse(
        predicted_label=res["predicted_label"],
        predicted_label_display=display_name(res["predicted_label"]),
        confidence=res["confidence"],
        probabilities=res["probabilities"],
        ambiguous=res["ambiguous"],
        model_id=request.model,
        model_version=meta["version"],
        rhetorical_section=section,
        passages_used=ks.status == key_sentences.USED,
        key_sentences={
            "status": ks.status,
            "source": ks.source,
            "detail": ks.detail,
            "cited_paper": {"id": ks.paper_id, "title": ks.title} if (ks.paper_id or ks.title) else None,
            "sentences": [{"rank": p.rank, "text": p.text, "similarity": p.similarity} for p in ks.sentences],
        },
        input_text=input_text,
        latency_ms={
            "cited_paper": ks.fetch_ms,
            "key_sentences": ks.encode_ms,
            "inference": res["latency_ms"],
            "total": total_ms,
        },
        status="success",
    )


# ── recuperación de pasajes ──────────────────────────────────────────────────

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
