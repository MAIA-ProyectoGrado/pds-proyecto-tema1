"""Registro de modelos servidos por la API, con carga diferida.

Ruta de los artefactos y modelo por defecto: ver `src/api/config.py`.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.passages import SentenceEncoder  # noqa: E402
from models.scif_predict import Predictor  # noqa: E402
from src.api import config  # noqa: E402

MODELS_DIR = config.MODELS_DIR
ENCODER_DIRNAME = "minilm-l6-v2"


class ModelNotFound(KeyError):
    """El id de modelo solicitado no existe en el registro."""


class ModelUnavailable(RuntimeError):
    """El modelo existe en el registro pero sus pesos no están en disco."""


# ── registro ─────────────────────────────────────────────────────────────────
# Los ids son estables entre versiones; `version` dice qué artefacto se sirve.
REGISTRY: dict[str, dict[str, Any]] = {
    "scibert": {
        "id": "scibert",
        "name": "SciBERT",
        "kind": "Encoder científico ajustado · 110 M parámetros",
        "reads": "Contexto, sección y 3 oraciones clave del abstract del artículo citado",
        "reads_passages": True,
        "version": "v4-config-b",
        "dirname": "scif-scibert-chunks",
        "recommended": True,
    },
    "tfidf-logreg": {
        "id": "tfidf-logreg",
        "name": "TF-IDF + Regresión Logística",
        "kind": "Línea base · scikit-learn",
        "reads": "Contexto y sección",
        "reads_passages": False,
        "version": "v4-tfidf",
        "dirname": "scif-v4-tfidf-logreg",
        "recommended": False,
    },
}

DEFAULT_MODEL = config.DEFAULT_MODEL
if DEFAULT_MODEL not in REGISTRY:
    raise config.ConfigError(
        f"SCIF_DEFAULT_MODEL={DEFAULT_MODEL!r} no existe. Opciones: {', '.join(REGISTRY)}."
    )

_cache: dict[str, Predictor] = {}
_locks: dict[str, threading.Lock] = {mid: threading.Lock() for mid in REGISTRY}
#: serializa la inferencia con torch (clasificador, MiniLM y embeddings de la
#: recuperación) para que los hilos no compitan por los núcleos
_infer_lock = threading.Lock()
_encoder = SentenceEncoder(MODELS_DIR / ENCODER_DIRNAME)

try:
    import torch

    # Con la inferencia serializada, torch puede usar varios hilos sin contención.
    torch.set_num_threads(config.TORCH_THREADS)
except Exception:
    pass


def model_dir(model_id: str) -> Path:
    if model_id not in REGISTRY:
        raise ModelNotFound(model_id)
    return MODELS_DIR / REGISTRY[model_id]["dirname"]


def is_available(model_id: str) -> bool:
    """Hay pesos en disco para este modelo."""
    d = model_dir(model_id)
    return (d / "model.pkl").exists() or (d / "config.json").exists()


def get_predictor(model_id: str) -> Predictor:
    """Carga (una sola vez) y devuelve el predictor del modelo pedido."""
    if model_id not in REGISTRY:
        raise ModelNotFound(model_id)
    if model_id in _cache:
        return _cache[model_id]
    with _locks[model_id]:
        if model_id in _cache:
            return _cache[model_id]
        if not is_available(model_id):
            raise ModelUnavailable(
                f"No hay pesos en {model_dir(model_id)}. "
                "Ejecuta `dvc pull` o define SCIF_MODELS_DIR."
            )
        _cache[model_id] = Predictor(model_dir(model_id), config.AMBIGUITY_THRESHOLD)
    return _cache[model_id]


# ── codificador de oraciones clave (MiniLM) ──────────────────────────────────

def encoder() -> SentenceEncoder:
    return _encoder


def encoder_available() -> bool:
    return _encoder.available


def encoder_loaded() -> bool:
    return _encoder._loaded


# ── inferencia ───────────────────────────────────────────────────────────────

def predict(citation_context: str, rhetorical_section: str | None, model_id: str,
            passages: list[str] | None = None, cited_title: str | None = None) -> dict:
    """Inferencia real. Devuelve además la entrada efectiva y la latencia medida."""
    predictor = get_predictor(model_id)
    started = time.perf_counter()
    with _infer_lock:
        result = predictor.predict(citation_context, rhetorical_section, passages, cited_title)
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    result["model_id"] = model_id
    return result


def catalog() -> list[dict[str, Any]]:
    """Metadatos de los modelos para poblar el selector del tablero."""
    out = []
    for mid, meta in REGISTRY.items():
        entry = {k: v for k, v in meta.items() if k != "dirname"}
        entry["available"] = is_available(mid)
        entry["loaded"] = mid in _cache
        out.append(entry)
    return out


def warm_up(model_id: str = DEFAULT_MODEL) -> None:
    """Precarga en segundo plano: SciBERT y MiniLM tardan varios segundos y sin
    esto la primera petición del tablero parece colgada. Se desactiva con
    SCIF_WARMUP=false."""
    if not config.WARMUP:
        return

    def _load() -> None:
        try:
            p = get_predictor(model_id)
            # Una pasada de prueba: la primera inferencia de torch tarda ~1 s más que
            # las siguientes, y así no la paga el primer usuario.
            with _infer_lock:
                p.predict("Warm-up citation context [1].", "Methods", ["Warm-up sentence."])
                if p.reads_passages and _encoder.available:
                    _encoder.encode(["Warm-up citation context [1].", "Warm-up sentence."])
        except Exception:
            pass

    threading.Thread(target=_load, name=f"warmup-{model_id}", daemon=True).start()
