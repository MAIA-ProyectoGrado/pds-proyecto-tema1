"""Pruebas de la API de inferencia.

Las de la línea base (TF-IDF v4, versionado en git) corren en el CI. Las de
SciBERT y MiniLM se omiten si faltan torch o los pesos (vienen de DVC).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from src.api import config, key_sentences, main, model_loader, retrieval  # noqa: E402
from src.api.labels import CANONICAL_LABELS, SECTION_OPTIONS  # noqa: E402
from src.api.limits import RateLimiter  # noqa: E402
from src.api.main import app  # noqa: E402

BASELINE = "tfidf-logreg"

baseline_required = pytest.mark.skipif(
    not model_loader.is_available(BASELINE),
    reason="faltan los pesos de la línea base (models/scif-v4-tfidf-logreg)",
)
scibert_required = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or not (model_loader.is_available("scibert") and model_loader.encoder_available()),
    reason="faltan torch o los pesos de SciBERT o de MiniLM (DVC)",
)


@pytest.mark.skipif(not model_loader.is_available("scibert"),
                    reason="sin los pesos de SciBERT no hace falta PyTorch")
def test_torch_is_installed_when_scibert_weights_are_present():
    assert importlib.util.find_spec("torch") is not None, (
        "Los pesos de SciBERT están descargados pero falta PyTorch: "
        "instala src/api/requirements.txt")


@pytest.fixture(scope="module")
def client():
    with fastapi_testclient.TestClient(app) as c:
        yield c


def post(client, context, section="Introduction", model=BASELINE, **extra):
    return client.post(
        "/predict",
        json={"citation_context": context, "rhetorical_section": section, "model": model, **extra},
    )


def no_network(*_a, **_k):
    raise AssertionError("no debería consultar arXiv")


# ── contrato básico ──────────────────────────────────────────────────────────

def test_root_ok(client):
    assert client.get("/").json()["status"] == "ok"


def test_health_reports_components_and_config(client):
    body = client.get("/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert set(body["models"]) == set(model_loader.REGISTRY)
    assert body["default_model"] == model_loader.DEFAULT_MODEL
    assert body["config"]["rate_limit_per_min"] == config.RATE_LIMIT_PER_MIN
    assert {"available", "loaded"} <= set(body["key_sentences_encoder"])


def test_models_catalog(client):
    body = client.get("/models").json()
    ids = {m["id"] for m in body["models"]}
    assert {"scibert", BASELINE} <= ids
    assert body["labels"] == CANONICAL_LABELS
    assert body["default_model"] in ids
    assert body["sections"] == SECTION_OPTIONS
    assert "Abstract" not in body["sections"]          # no aparece en el entrenamiento


def test_catalog_explains_what_each_model_reads(client):
    models = {m["id"]: m for m in client.get("/models").json()["models"]}
    assert models["scibert"]["reads_passages"] is True
    assert models[BASELINE]["reads_passages"] is False
    assert all(m["reads"] for m in models.values())
    assert all(m["version"].startswith("v4") for m in models.values() if m["backend"] == "local")


def test_catalog_does_not_publish_metrics(client):
    """El catálogo describe los modelos servibles, no su desempeño: las métricas
    van en el reporte, donde se puede explicar su procedencia."""
    scibert = next(m for m in client.get("/models").json()["models"] if m["id"] == "scibert")
    assert scibert["recommended"] is True
    assert scibert["available"] is model_loader.is_available("scibert")
    assert not any(k.startswith("f1") or k.startswith("metric") for k in scibert)


def _without_scibert(monkeypatch):
    real = model_loader.is_available
    monkeypatch.setattr(model_loader, "is_available", lambda mid: mid != "scibert" and real(mid))


@baseline_required
def test_catalog_explains_missing_weights(client, monkeypatch):
    _without_scibert(monkeypatch)
    models = {m["id"]: m for m in client.get("/models").json()["models"]}
    reason = models["scibert"]["unavailable_reason"]
    assert models["scibert"]["available"] is False
    assert "models/scif-scibert-chunks" in reason and "README" in reason
    assert models[BASELINE]["available"] is True and models[BASELINE]["unavailable_reason"] is None


def test_predict_without_weights_says_how_to_get_them(client, monkeypatch):
    _without_scibert(monkeypatch)
    r = post(client, "We fine-tune SciBERT [4].", model="scibert")
    assert r.status_code == 503
    assert r.json()["detail"] == model_loader.missing_weights_message("scibert")


def test_docs_are_served(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").json()["info"]["version"] == main.API_VERSION


def test_empty_context_is_rejected(client):
    assert client.post("/predict", json={"citation_context": "   "}).status_code == 400


def test_unknown_model_is_rejected(client):
    r = client.post("/predict", json={"citation_context": "x", "model": "gpt-4o"})
    assert r.status_code == 422


# ── uso controlado ───────────────────────────────────────────────────────────

def test_context_longer_than_limit_is_rejected_with_readable_message(client):
    r = client.post("/predict", json={"citation_context": "a" * (config.MAX_CONTEXT_CHARS + 1)})
    assert r.status_code == 422
    assert r.json()["detail"] == (
        f"El campo citation_context admite como máximo {config.MAX_CONTEXT_CHARS} caracteres."
    )


def test_body_larger_than_limit_is_413(client):
    big = b'{"citation_context": "' + b"a" * (config.MAX_BODY_KB * 1024) + b'"}'
    r = client.post("/predict", content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(main, "limiter", RateLimiter(2))
    codes = [client.post("/predict", json={"citation_context": " "}).status_code for _ in range(3)]
    assert codes == [400, 400, 429]
    r = client.post("/retrieve", json={"citation_context": "x"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert client.get("/health").status_code == 200   # las lecturas no cuentan


def test_rate_limiter_window_slides():
    rl = RateLimiter(2, window_s=60)
    assert [rl.hit("ip", now=t) for t in (0, 1)] == [0, 0]
    assert rl.hit("ip", now=30) == 30                 # el cupo más viejo se libera en t=60
    assert rl.hit("otra-ip", now=30) == 0             # la cuenta es por IP
    assert rl.hit("ip", now=60) == 0
    assert RateLimiter(0).hit("ip") == 0              # 0 desactiva el límite


# ── la línea base: predicciones reales ───────────────────────────────────────

@baseline_required
def test_prediction_is_deterministic(client):
    ctx = "For a comprehensive review, see Jones (2019)."
    first = post(client, ctx).json()
    second = post(client, ctx).json()
    assert first["probabilities"] == second["probabilities"]
    assert first["predicted_label"] == second["predicted_label"]


@baseline_required
def test_model_discriminates_between_classes(client):
    labels = {
        post(client, "Although several approaches exist, it remains an open problem [3].").json()["predicted_label"],
        post(client, "We use the ADAM optimizer (Kingma & Ba, 2015).").json()["predicted_label"],
        post(client, "For a comprehensive review, see Jones (2019).").json()["predicted_label"],
    }
    assert len(labels) > 1, f"el clasificador colapsó a una sola clase: {labels}"


@baseline_required
def test_rhetorical_section_reaches_the_model(client):
    ctx = "Our results are in strong agreement with prior work."
    intro = post(client, ctx, section="Introduction").json()
    disc = post(client, ctx, section="Discussion").json()
    assert intro["input_text"].endswith("[SEC] Introduccion")
    assert disc["input_text"].endswith("[SEC] Discusion")
    assert intro["probabilities"] != disc["probabilities"]


@baseline_required
@pytest.mark.parametrize("section", [None, "", "   "])
def test_missing_section_is_sent_as_unknown(client, section):
    body = post(client, "We follow the protocol of Smith et al. (2020).", section=section).json()
    assert body["rhetorical_section"] is None
    assert body["input_text"].endswith("[SEC] Desconocida")


@baseline_required
def test_baseline_ignores_the_cited_paper_without_calling_arxiv(client, monkeypatch):
    monkeypatch.setattr(retrieval, "get_arxiv_metadata", no_network)
    body = post(client, "We use the ADAM optimizer (Kingma & Ba, 2015).", arxiv_id="1412.6980").json()
    assert body["passages_used"] is False
    assert body["key_sentences"]["status"] == key_sentences.NOT_READ_BY_MODEL
    assert body["key_sentences"]["sentences"] == []


@baseline_required
def test_response_shape(client):
    body = post(client, "We adopt the evaluation protocol of Smith et al. (2020).").json()
    assert set(body["probabilities"]) == set(CANONICAL_LABELS)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.02
    assert body["predicted_label"] in CANONICAL_LABELS
    assert body["model_id"] == BASELINE
    assert body["model_version"] == "v4-tfidf"
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["ambiguous"], bool)
    lat = body["latency_ms"]
    assert set(lat) == {"cited_paper", "key_sentences", "inference", "total"}
    assert lat["total"] >= lat["inference"] >= 0


@baseline_required
def test_confidence_matches_top_probability(client):
    body = post(client, "Unlike previous approaches, our method is single-stage.").json()
    assert body["confidence"] == max(body["probabilities"].values())


# ── oraciones clave: los casos sin modelo ni red ─────────────────────────────

CTX = "We build our encoder on top of SciBERT [4]."


def test_key_sentences_not_provided():
    ks = key_sentences.build(CTX, reads_passages=True)
    assert ks.status == key_sentences.NOT_PROVIDED and ks.sentences == []


def test_key_sentences_invalid_id_does_not_fail(monkeypatch):
    monkeypatch.setattr(model_loader, "encoder_available", lambda: True)
    ks = key_sentences.build(CTX, reads_passages=True, arxiv_id="hola")
    assert ks.status == key_sentences.FAILED
    assert "no es un identificador de arXiv válido" in ks.detail


@pytest.mark.parametrize("error,fragment", [
    (retrieval.PaperNotFound("2401.00002"), "no tiene ningún artículo"),
    (retrieval.DownloadError("No se pudo consultar arXiv: timeout."), "No se pudo consultar arXiv"),
])
def test_key_sentences_arxiv_failures_are_explained(monkeypatch, error, fragment):
    monkeypatch.setattr(model_loader, "encoder_available", lambda: True)

    def boom(_id):
        raise error

    monkeypatch.setattr(retrieval, "get_arxiv_metadata", boom)
    ks = key_sentences.build(CTX, reads_passages=True, arxiv_id="2401.00002")
    assert ks.status == key_sentences.FAILED
    assert fragment in ks.detail and ks.detail.endswith("El modelo clasificó sin el artículo citado.")


def test_key_sentences_without_encoder_are_explained(monkeypatch):
    monkeypatch.setattr(model_loader, "encoder_available", lambda: False)
    ks = key_sentences.build(CTX, reads_passages=True, arxiv_id="1903.10676")
    assert ks.status == key_sentences.FAILED and "minilm" in ks.detail


# ── SciBERT configuración B (opcional) ───────────────────────────────────────

ABSTRACT = ("We release SciBERT, a pretrained language model based on BERT. "
            "It is trained on a large corpus of scientific publications. "
            "We evaluate on sequence tagging, sentence classification and dependency parsing. "
            "The vocabulary is built from scientific text with WordPiece. "
            "Results improve over BERT on most tasks.")


@pytest.fixture
def fake_arxiv(monkeypatch):
    retrieval._meta_cache.clear()

    def fake_meta(arxiv_id):
        return {"id": arxiv_id, "title": "SciBERT", "authors": [], "year": "2019",
                "abstract": ABSTRACT, "source": "arxiv-pdf"}

    monkeypatch.setattr(retrieval, "fetch_arxiv_metadata", fake_meta)
    yield
    retrieval._meta_cache.clear()


@scibert_required
def test_scibert_reads_key_sentences_from_arxiv_abstract(client, fake_arxiv):
    body = post(client, CTX, section="Methods", model="scibert", arxiv_id="1903.10676").json()
    ks = body["key_sentences"]
    assert body["model_version"] == "v4-config-b"
    assert body["passages_used"] is True and ks["status"] == "used" and ks["source"] == "arxiv-abstract"
    assert ks["cited_paper"] == {"id": "1903.10676", "title": "SciBERT"}
    assert len(ks["sentences"]) == 3
    sims = [s["similarity"] for s in ks["sentences"]]
    assert sims == sorted(sims, reverse=True)
    assert all(s["text"] in ABSTRACT for s in ks["sentences"])
    expected_b = "Section: Methods | Cited passages: " + " ".join(s["text"] for s in ks["sentences"])
    assert body["input_text"] == f"{CTX} [SEP] {expected_b}"


@scibert_required
def test_scibert_without_cited_paper_uses_training_fallback(client):
    body = post(client, CTX, section=None, model="scibert").json()
    assert body["passages_used"] is False
    assert body["key_sentences"]["status"] == "not_provided"
    assert body["input_text"] == f"{CTX} [SEP] Passages unavailable"


@scibert_required
def test_scibert_with_pasted_text(client):
    body = post(client, CTX, section="Methods", model="scibert", cited_text=ABSTRACT).json()
    assert body["passages_used"] is True
    assert body["key_sentences"]["source"] == "text"
    assert body["key_sentences"]["cited_paper"] is None


@scibert_required
def test_scibert_falls_back_when_arxiv_fails(client, monkeypatch):
    retrieval._meta_cache.clear()

    def boom(_id):
        raise retrieval.DownloadError("No se pudo consultar arXiv: timeout.")

    monkeypatch.setattr(retrieval, "fetch_arxiv_metadata", boom)
    r = post(client, CTX, section="Methods", model="scibert", arxiv_id="1903.10676")
    assert r.status_code == 200
    body = r.json()
    assert body["passages_used"] is False and body["key_sentences"]["status"] == "failed"
    assert body["input_text"].endswith("Section: Methods | Passages unavailable")
