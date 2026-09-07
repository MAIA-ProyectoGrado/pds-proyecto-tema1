"""Registro de modelos servidos por la API, con lazy load

Ruta de los artefactos: `SCIF_MODELS_DIR` (por defecto `<repo>/models`).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.scif_predict import Predictor, build_input  # noqa: E402

MODELS_DIR = Path(os.environ.get("SCIF_MODELS_DIR") or (REPO_ROOT / "models"))


class ModelNotFound(KeyError):
    """El id de modelo solicitado no existe en el registro."""


class ModelUnavailable(RuntimeError):
    """El modelo existe en el registro pero sus pesos no están en disco."""


# ── Registro ──
REGISTRY: dict[str, dict[str, Any]] = {
    "scibert": {
        "id": "scibert",
        "name": "SciBERT fine-tuned",
        "kind": "Encoder de dominio científico · 110M",
        "dirname": "scif-scibert",
        "recommended": True,
        "f1_macro_val": None,
        "metric_verified": False,
        "metric_note": (
            ""
        ),
    },
    "tfidf-logreg": {
        "id": "tfidf-logreg",
        "name": "TF-IDF + Regresión Logística",
        "kind": "Línea base · sklearn",
        "dirname": "scif-v1-tfidf-logreg",
        "recommended": False,
        "f1_macro_val": 0.516,
        "metric_verified": False,
        "metric_note": (
            ""
        ),
    },
}

DEFAULT_MODEL = "scibert"

_cache: dict[str, Predictor] = {}
_locks: dict[str, threading.Lock] = {mid: threading.Lock() for mid in REGISTRY}
_infer_lock = threading.Lock()

try:  
    import torch

    torch.set_num_threads(1)
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
        _cache[model_id] = Predictor(model_dir(model_id))
    return _cache[model_id]


def predict(citation_context: str, rhetorical_section: str | None, model_id: str) -> dict:
    """Inferencia real. Devuelve además la entrada efectiva y la latencia medida."""
    predictor = get_predictor(model_id)
    text = build_input(citation_context, rhetorical_section)
    started = time.perf_counter()
    with _infer_lock:
        result = predictor.predict(citation_context, rhetorical_section)
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    result["model_id"] = model_id
    result["input_text"] = text
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
    """Precarga en segundo plano: SciBERT tarda ~10-15 s y sin esto la primera
    petición del tablero parece colgada."""

    def _load() -> None:
        try:
            get_predictor(model_id)
        except Exception: 
            pass

    threading.Thread(target=_load, name=f"warmup-{model_id}", daemon=True).start()
