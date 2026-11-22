"""Parámetros de funcionamiento de la API, leídos de variables de entorno.

Todas son opcionales: sin ninguna definida el servicio arranca con los valores
por defecto de la tabla. Un valor inválido detiene el arranque con un mensaje
que nombra la variable, en lugar de ignorarse en silencio.

| Variable                     | Defecto                                     |
|------------------------------|---------------------------------------------|
| SCIF_MODELS_DIR              | <repo>/models                               |
| SCIF_DEFAULT_MODEL           | scibert                                     |
| SCIF_WARMUP                  | true                                        |
| SCIF_TORCH_THREADS           | min(4, núcleos)                             |
| SCIF_WORKERS                 | 1 (lo lee el comando de arranque)           |
| SCIF_AMBIGUITY_THRESHOLD     | 0.10                                        |
| SCIF_TOP_K                   | 3                                           |
| SCIF_CHUNK_WORDS             | 300                                         |
| SCIF_DOC_CACHE_SIZE          | 32                                          |
| SCIF_HTTP_TIMEOUT            | 40                                          |
| SCIF_MAX_PDF_MB              | 25                                          |
| SCIF_MAX_PDF_PAGES           | 40                                          |
| SCIF_MAX_CONTEXT_CHARS       | 3000                                        |
| SCIF_MAX_CITED_TEXT_CHARS    | 200000                                      |
| SCIF_MAX_BODY_KB             | 1024                                        |
| SCIF_RATE_LIMIT_PER_MIN      | 30                                          |
| SCIF_ROOT_PATH               | (vacío)                                     |
| SCIF_ALLOWED_ORIGINS         | http://localhost:8080,http://127.0.0.1:8080 |

La descripción de cada una está en `.env.example`.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Una variable de entorno tiene un valor no válido."""


def _raw(name: str) -> str | None:
    v = os.environ.get(name)
    return v.strip() if v is not None and v.strip() != "" else None


def _int(name: str, default: int, lo: int, hi: int) -> int:
    v = _raw(name)
    if v is None:
        return default
    try:
        n = int(v)
    except ValueError:
        raise ConfigError(f"{name}={v!r} no es un entero.") from None
    if not lo <= n <= hi:
        raise ConfigError(f"{name}={n} fuera de rango; debe estar entre {lo} y {hi}.")
    return n


def _float(name: str, default: float, lo: float, hi: float) -> float:
    v = _raw(name)
    if v is None:
        return default
    try:
        x = float(v.replace(",", "."))
    except ValueError:
        raise ConfigError(f"{name}={v!r} no es un número.") from None
    if not lo <= x <= hi:
        raise ConfigError(f"{name}={x} fuera de rango; debe estar entre {lo} y {hi}.")
    return x


def _bool(name: str, default: bool) -> bool:
    v = _raw(name)
    if v is None:
        return default
    if v.lower() in {"1", "true", "yes", "si", "sí", "on"}:
        return True
    if v.lower() in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name}={v!r} debe ser true o false.")


def _path_prefix(name: str) -> str:
    v = _raw(name)
    if v is None:
        return ""
    if not v.startswith("/") or v.endswith("/") or " " in v:
        raise ConfigError(f"{name}={v!r} debe empezar con '/' y no terminar en '/' (por ejemplo /api).")
    return v


# ── modelos ──────────────────────────────────────────────────────────────────
MODELS_DIR = Path(_raw("SCIF_MODELS_DIR") or (REPO_ROOT / "models"))
#: se valida contra el registro en model_loader, que es quien conoce los ids
DEFAULT_MODEL = _raw("SCIF_DEFAULT_MODEL") or "scibert"
WARMUP = _bool("SCIF_WARMUP", True)
TORCH_THREADS = _int("SCIF_TORCH_THREADS", min(4, os.cpu_count() or 1), 1, 64)
#: procesos de uvicorn; lo usa el comando de arranque, aquí solo se informa
WORKERS = _int("SCIF_WORKERS", 1, 1, 16)

# ── clasificación ────────────────────────────────────────────────────────────
#: margen mínimo entre las dos clases más probables para no marcar ambigüedad
AMBIGUITY_THRESHOLD = _float("SCIF_AMBIGUITY_THRESHOLD", 0.10, 0.0, 1.0)

# ── artículo citado y recuperación de pasajes ───────────────────────────────
TOP_K = _int("SCIF_TOP_K", 3, 1, 10)
CHUNK_WORDS = _int("SCIF_CHUNK_WORDS", 300, 50, 1000)
DOC_CACHE_SIZE = _int("SCIF_DOC_CACHE_SIZE", 32, 0, 1000)
HTTP_TIMEOUT = _int("SCIF_HTTP_TIMEOUT", 40, 5, 300)
MAX_PDF_MB = _int("SCIF_MAX_PDF_MB", 25, 1, 200)
MAX_PDF_PAGES = _int("SCIF_MAX_PDF_PAGES", 40, 1, 500)

# ── uso controlado ───────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS = _int("SCIF_MAX_CONTEXT_CHARS", 3000, 100, 20000)
MAX_CITED_TEXT_CHARS = _int("SCIF_MAX_CITED_TEXT_CHARS", 200_000, 1000, 2_000_000)
MAX_BODY_KB = _int("SCIF_MAX_BODY_KB", 1024, 16, 16384)
#: peticiones POST por minuto y por IP; 0 desactiva el límite
RATE_LIMIT_PER_MIN = _int("SCIF_RATE_LIMIT_PER_MIN", 30, 0, 10000)

# ── red ──────────────────────────────────────────────────────────────────────
#: prefijo bajo el que un proxy publica la API (en Docker, /api); hace que /docs
#: pida el esquema en la ruta correcta
ROOT_PATH = _path_prefix("SCIF_ROOT_PATH")
ALLOWED_ORIGINS = [
    o.strip()
    for o in (_raw("SCIF_ALLOWED_ORIGINS") or "http://localhost:8080,http://127.0.0.1:8080").split(",")
    if o.strip()
]


def summary() -> dict:
    """Valores efectivos, para /health. No incluye nada sensible."""
    return {
        "models_dir": str(MODELS_DIR),
        "default_model": DEFAULT_MODEL,
        "warmup": WARMUP,
        "torch_threads": TORCH_THREADS,
        "workers": WORKERS,
        "ambiguity_threshold": AMBIGUITY_THRESHOLD,
        "top_k": TOP_K,
        "chunk_words": CHUNK_WORDS,
        "doc_cache_size": DOC_CACHE_SIZE,
        "http_timeout_s": HTTP_TIMEOUT,
        "max_pdf_mb": MAX_PDF_MB,
        "max_pdf_pages": MAX_PDF_PAGES,
        "max_context_chars": MAX_CONTEXT_CHARS,
        "max_cited_text_chars": MAX_CITED_TEXT_CHARS,
        "max_body_kb": MAX_BODY_KB,
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
        "root_path": ROOT_PATH,
    }
