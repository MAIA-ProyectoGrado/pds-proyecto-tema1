"""Pruebas de la API de inferencia.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from src.api import model_loader  # noqa: E402
from src.api.labels import CANONICAL_LABELS  # noqa: E402
from src.api.main import app  # noqa: E402

BASELINE = "tfidf-logreg"

baseline_required = pytest.mark.skipif(
    not model_loader.is_available(BASELINE),
    reason="faltan los pesos de la línea base (models/scif-v1-tfidf-logreg)",
)


@pytest.fixture(scope="module")
def client():
    with fastapi_testclient.TestClient(app) as c:
        yield c


def post(client, context, section="Introduction", model=BASELINE):
    return client.post(
        "/predict",
        json={"citation_context": context, "rhetorical_section": section, "model": model},
    )


# ── contrato básico ──────────────────────────────────────────────────────────

def test_root_ok(client):
    assert client.get("/").json()["status"] == "ok"


def test_models_catalog(client):
    body = client.get("/models").json()
    ids = {m["id"] for m in body["models"]}
    assert {"scibert", BASELINE} <= ids
    assert body["labels"] == CANONICAL_LABELS
    assert body["default_model"] in ids


def test_scibert_metric_is_not_published_as_verified(client):
    """El artefacto servido se empaquetó a 1 época y el 0.640 documentado es de 4.
    Mientras no se pueda re-medir, la API no debe presentarlo como dato firme."""
    scibert = next(m for m in client.get("/models").json()["models"] if m["id"] == "scibert")
    assert scibert["metric_verified"] is False
    assert scibert["f1_macro_val"] is None


def test_empty_context_is_rejected(client):
    assert client.post("/predict", json={"citation_context": "   "}).status_code == 400


def test_unknown_model_is_rejected(client):
    r = client.post("/predict", json={"citation_context": "x", "model": "gpt-4o"})
    assert r.status_code == 422


# ── el stub murió ────────────────────────────────────────────────────────────

@baseline_required
def test_prediction_is_deterministic(client):
    """El stub devolvía random.uniform(); un modelo real repite la misma salida."""
    ctx = "For a comprehensive review, see Jones (2019)."
    first = post(client, ctx).json()
    second = post(client, ctx).json()
    assert first["probabilities"] == second["probabilities"]
    assert first["predicted_label"] == second["predicted_label"]


@baseline_required
def test_model_discriminates_between_classes(client):
    """El stub devolvía siempre 'Gap'. Un modelo real separa contextos distintos."""
    labels = {
        post(client, "Although several approaches exist, it remains an open problem [3].").json()["predicted_label"],
        post(client, "We use the ADAM optimizer (Kingma & Ba, 2015).").json()["predicted_label"],
        post(client, "For a comprehensive review, see Jones (2019).").json()["predicted_label"],
    }
    assert len(labels) > 1, f"el clasificador colapsó a una sola clase: {labels}"


@baseline_required
def test_rhetorical_section_reaches_the_model(client):
    """`rhetorical_section` se descartaba en main.py; debe llegar a la entrada."""
    ctx = "Our results are in strong agreement with prior work."
    intro = post(client, ctx, section="Introduction").json()
    disc = post(client, ctx, section="Discussion").json()
    assert intro["input_text"].endswith("[SEC] Introduccion")
    assert disc["input_text"].endswith("[SEC] Discusion")
    assert intro["probabilities"] != disc["probabilities"]


# ── forma de la respuesta ────────────────────────────────────────────────────

@baseline_required
def test_response_shape(client):
    body = post(client, "We adopt the evaluation protocol of Smith et al. (2020).").json()
    assert set(body["probabilities"]) == set(CANONICAL_LABELS)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.02
    assert body["predicted_label"] in CANONICAL_LABELS
    assert body["model_id"] == BASELINE
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["ambiguous"], bool)
    assert body["latency_ms"] >= 0


@baseline_required
def test_confidence_matches_top_probability(client):
    body = post(client, "Unlike previous approaches, our method is single-stage.").json()
    assert body["confidence"] == max(body["probabilities"].values())


# ── SciBERT (opcional) ───────────────────────────────────────────────────────

@pytest.mark.skipif(
    not model_loader.is_available("scibert"), reason="faltan los pesos de SciBERT (DVC)"
)
def test_scibert_serves_predictions(client):
    pytest.importorskip("torch")
    body = post(client, "For a comprehensive review, see Jones (2019).", model="scibert").json()
    assert body["model_id"] == "scibert"
    assert body["predicted_label"] in CANONICAL_LABELS
