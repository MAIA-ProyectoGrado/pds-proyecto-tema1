"""Pruebas de los modelos de lenguaje en Bedrock (`src/api/llm.py` y su ruta en /predict).

Usan un cliente de Bedrock simulado: no hacen llamadas reales ni tienen costo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from src.api import config, key_sentences, llm, main, model_loader, retrieval  # noqa: E402
from src.api.labels import CANONICAL_LABELS  # noqa: E402
from src.api.limits import RateLimiter  # noqa: E402
from src.api.main import app  # noqa: E402
from src.api.prompts_llm import ORDEN  # noqa: E402

LLMS = ["nova-pro", "ministral-3-8b"]
CTX = "We build our encoder on top of SciBERT [4]."
BASIS = "[0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"     # Basis es la tercera en ORDEN


class FakeBedrock:
    """Imita `converse` de bedrock-runtime y guarda las peticiones."""

    def __init__(self, text=BASIS, tokens_in=2000, tokens_out=40, error=None):
        self.text, self.tokens_in, self.tokens_out, self.error = text, tokens_in, tokens_out, error
        self.calls = []

    def converse(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return {"output": {"message": {"content": [{"text": self.text}]}},
                "usage": {"inputTokens": self.tokens_in, "outputTokens": self.tokens_out},
                "stopReason": "end_turn"}


class AccessDenied(Exception):
    response = {"Error": {"Code": "AccessDeniedException",
                          "Message": "User arn:aws:sts::123456789012:assumed-role/x is not authorized"}}


@pytest.fixture(scope="module")
def client():
    with fastapi_testclient.TestClient(app) as c:
        yield c


@pytest.fixture
def bedrock(monkeypatch, tmp_path):
    """Bedrock activado con un cliente simulado y un presupuesto en un archivo temporal."""
    fake = FakeBedrock()
    monkeypatch.setattr(config, "LLM_ENABLED", True)
    monkeypatch.setattr(llm, "_boto3_installed", lambda: True)
    monkeypatch.setattr(llm, "_client", fake)
    monkeypatch.setattr(llm, "budget", llm.DailyBudget(tmp_path / "gasto.json", 5.0))
    monkeypatch.setattr(main, "llm_limiter", RateLimiter(0))
    return fake


def post(client, model="nova-pro", context=CTX, section="Methods", **extra):
    return client.post("/predict", json={"citation_context": context, "rhetorical_section": section,
                                         "model": model, **extra})


# ── catálogo y disponibilidad ────────────────────────────────────────────────

def test_catalog_lists_the_two_bedrock_models(client):
    models = {m["id"]: m for m in client.get("/models").json()["models"]}
    for mid in LLMS:
        m = models[mid]
        assert m["backend"] == "bedrock" and m["recommended"] is False
        assert m["cost_per_query_usd"] > 0 and m["typical_latency_ms"] > 0
        assert "abstract" in m["reads"]
    assert models["scibert"]["backend"] == "local" and models["scibert"]["cost_per_query_usd"] is None


def test_disabled_by_default_and_explains_why(client, monkeypatch):
    monkeypatch.setattr(config, "LLM_ENABLED", False)
    models = {m["id"]: m for m in client.get("/models").json()["models"]}
    assert models["nova-pro"]["available"] is False
    assert "SCIF_LLM_ENABLED" in models["nova-pro"]["unavailable_reason"]
    r = post(client)
    assert r.status_code == 503 and "SCIF_LLM_ENABLED" in r.json()["detail"]


def test_available_when_enabled(client, bedrock):
    models = {m["id"]: m for m in client.get("/models").json()["models"]}
    assert all(models[mid]["available"] and models[mid]["loaded"] for mid in LLMS)


def test_health_reports_budget_and_llm_config(client, bedrock):
    body = client.get("/health").json()
    assert body["llm_budget"]["budget_usd"] == 5.0 and body["llm_budget"]["spent_usd"] == 0
    assert {"llm_enabled", "bedrock_region", "llm_daily_budget_usd", "llm_rate_limit_per_min"} <= set(body["config"])
    assert set(LLMS) <= set(body["models"])


def test_bedrock_models_have_no_weights_folder():
    with pytest.raises(model_loader.ModelNotFound):
        model_loader.model_dir("nova-pro")


# ── la consulta, igual que en la evaluación ──────────────────────────────────

@pytest.mark.parametrize("model", LLMS)
def test_request_matches_the_evaluation(client, bedrock, model):
    assert post(client, model=model).status_code == 200
    call = bedrock.calls[-1]
    assert call["modelId"] == llm.MODELS[model]["bedrock_id"]
    assert call["inferenceConfig"] == {"maxTokens": 80, "temperature": 0.0}
    system = call["system"][0]["text"]
    assert "Definitions and criteria" in system and "Solved examples" in system      # enfoque E4
    user = call["messages"][0]["content"][0]["text"]
    assert user.startswith(f'Citation context: "{CTX}"\nSection: Methods\n')


def test_response_shape_and_scores(client, bedrock):
    body = post(client).json()
    assert body["predicted_label"] == "Basis" and body["model_id"] == "nova-pro"
    assert set(body["probabilities"]) == set(CANONICAL_LABELS)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-9
    assert body["confidence"] == 0.9                         # el puntaje mayor, sin normalizar
    assert body["probabilities"]["Basis"] == pytest.approx(0.9 / 1.0)
    assert body["llm"]["scores"] == dict(zip(ORDEN, json.loads(BASIS)))
    assert body["llm"]["approach"] == "E4" and body["llm"]["raw_response"] == BASIS
    assert body["passages_used"] is False and body["key_sentences"]["sentences"] == []
    assert body["input_text"] == bedrock.calls[-1]["messages"][0]["content"][0]["text"]
    assert body["latency_ms"]["key_sentences"] == 0.0


def test_cost_comes_from_real_tokens_and_accumulates(client, bedrock):
    first = post(client).json()["llm"]
    expected = (2000 * 0.80 + 40 * 3.20) / 1e6
    assert first["cost_usd"] == pytest.approx(expected)
    second = post(client).json()["llm"]
    assert second["spent_today_usd"] == pytest.approx(2 * expected, abs=1e-4)
    assert llm.budget.spent() == pytest.approx(2 * expected)


def test_tie_is_marked_ambiguous(client, bedrock):
    bedrock.text = "[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
    body = post(client).json()
    assert body["ambiguous"] is True and body["llm"]["tie"] is True
    assert body["predicted_label"] == "Background"          # en empate, la primera en ORDEN


def test_ambiguity_uses_the_raw_score_margin(client, bedrock):
    bedrock.text = "[0.2, 0.0, 0.7, 0.1, 0.6, 0.0, 0.0, 0.0, 0.0]"     # margen 0,1: no es ambiguo
    assert post(client).json()["ambiguous"] is False
    bedrock.text = "[0.2, 0.0, 0.7, 0.1, 0.65, 0.0, 0.0, 0.0, 0.0]"
    assert post(client).json()["ambiguous"] is True


def test_all_zero_scores_give_uniform_distribution(client, bedrock):
    bedrock.text = "[" + ", ".join(["0.0"] * 9) + "]"
    body = post(client).json()
    assert all(v == pytest.approx(1 / 9) for v in body["probabilities"].values())
    assert body["ambiguous"] is True


# ── artículo citado ──────────────────────────────────────────────────────────

def test_without_cited_paper_the_prompt_says_not_available(client, bedrock):
    body = post(client).json()
    assert body["key_sentences"]["status"] == key_sentences.NOT_PROVIDED
    assert "Cited paper title: not available\nCited paper abstract: not available" in body["input_text"]


def test_reads_title_and_abstract_from_arxiv(client, bedrock, monkeypatch):
    meta = {"title": "SciBERT: A Pretrained Language Model", "abstract": "We release SciBERT, a model."}
    monkeypatch.setattr(retrieval, "get_arxiv_metadata", lambda pid: meta)
    body = post(client, arxiv_id="1903.10676").json()
    ks = body["key_sentences"]
    assert ks["status"] == key_sentences.ABSTRACT_USED and ks["source"] == "arxiv-abstract"
    assert ks["cited_paper"] == {"id": "1903.10676", "title": meta["title"]}
    assert ks["abstract"] == meta["abstract"]
    assert f"Cited paper title: {meta['title']}\nCited paper abstract: {meta['abstract']}" in body["input_text"]


def test_arxiv_failure_still_classifies(client, bedrock, monkeypatch):
    def down(pid):
        raise retrieval.DownloadError("arXiv no respondió.")
    monkeypatch.setattr(retrieval, "get_arxiv_metadata", down)
    body = post(client, arxiv_id="1903.10676").json()
    assert body["key_sentences"]["status"] == key_sentences.FAILED
    assert "clasificó sin el artículo citado" in body["key_sentences"]["detail"]
    assert body["predicted_label"] == "Basis"


def test_pasted_text_without_abstract_uses_first_words(client, bedrock):
    text = " ".join(f"w{i}" for i in range(400))
    body = post(client, cited_text=text).json()
    ks = body["key_sentences"]
    assert ks["status"] == key_sentences.ABSTRACT_USED and ks["source"] == "text"
    assert ks["abstract"].split() == text.split()[:key_sentences.LLM_TEXT_WORDS]
    assert "250 palabras" in ks["detail"]


def test_long_abstract_is_capped(bedrock):
    prompt = llm.user_prompt(CTX, None, None, "x" * 10_000)
    assert prompt.endswith("x" * llm.MAX_ABSTRACT_CHARS) and "x" * (llm.MAX_ABSTRACT_CHARS + 1) not in prompt


# ── errores ──────────────────────────────────────────────────────────────────

def test_invalid_answer_is_502_and_is_still_charged(client, bedrock):
    bedrock.text = "The answer is Basis."
    r = post(client)
    assert r.status_code == 502
    assert "9 puntajes" in r.json()["detail"] and "The answer is Basis." in r.json()["detail"]
    assert llm.budget.spent() == pytest.approx((2000 * 0.80 + 40 * 3.20) / 1e6)


def test_bedrock_error_is_explained_without_aws_details(client, bedrock):
    bedrock.error = AccessDenied()
    r = post(client)
    assert r.status_code == 502
    assert "bedrock:InvokeModel" in r.json()["detail"] and "123456789012" not in r.json()["detail"]
    assert llm.budget.spent() == 0.0                         # la reserva se libera


# ── topes ────────────────────────────────────────────────────────────────────

def test_daily_budget_blocks_before_calling_bedrock(client, bedrock, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "budget", llm.DailyBudget(tmp_path / "lleno.json", 0.001))
    r = post(client)
    assert r.status_code == 429
    assert "tope diario" in r.json()["detail"] and int(r.headers["Retry-After"]) > 0
    assert bedrock.calls == []


def test_budget_zero_blocks_everything(client, bedrock, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "budget", llm.DailyBudget(tmp_path / "cero.json", 0.0))
    assert post(client, model="ministral-3-8b").status_code == 429


def test_local_models_ignore_the_llm_budget(client, bedrock, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "budget", llm.DailyBudget(tmp_path / "cero.json", 0.0))
    if not model_loader.is_available("tfidf-logreg"):
        pytest.skip("faltan los pesos de la línea base")
    assert post(client, model="tfidf-logreg").status_code == 200


def test_llm_rate_limit_per_ip(client, bedrock, monkeypatch):
    monkeypatch.setattr(main, "llm_limiter", RateLimiter(1))
    assert post(client).status_code == 200
    r = post(client, model="ministral-3-8b")
    assert r.status_code == 429 and "consultas por minuto" in r.json()["detail"]
    assert "Retry-After" in r.headers and len(bedrock.calls) == 1


def test_budget_persists_and_resets_each_day(tmp_path):
    path = tmp_path / "gasto.json"
    day = ["2026-11-24"]
    b = llm.DailyBudget(path, 1.0, today=lambda: day[0])
    b.reserve(0.4)
    b.settle(0.4, 0.3)
    assert llm.DailyBudget(path, 1.0, today=lambda: day[0]).spent() == pytest.approx(0.3)   # otro proceso
    with pytest.raises(llm.BudgetExceeded):
        b.reserve(0.8)
    assert b.spent() == pytest.approx(0.3)                   # una reserva rechazada no suma
    day[0] = "2026-11-25"
    assert b.spent() == 0.0
    b.reserve(0.8)


def test_reservation_covers_the_worst_case():
    system, user = llm.system_prompt(), llm.user_prompt(CTX, "Methods", "A title", "An abstract.")
    real_tokens_in = (len(system) + len(user)) / 4                 # inglés: unos 4 caracteres por token
    assert llm.max_cost("nova-pro", system, user) >= llm.cost("nova-pro", int(real_tokens_in), llm.MAX_TOKENS)


def test_unwritable_spend_file_blocks_llms_but_not_health(client, bedrock, monkeypatch, tmp_path):
    blocker = tmp_path / "archivo"
    blocker.write_text("")
    monkeypatch.setattr(llm, "budget", llm.DailyBudget(blocker / "gasto.json", 5.0))   # su carpeta es un archivo
    assert client.get("/health").json()["llm_budget"]["spent_usd"] is None
    r = post(client)
    assert r.status_code == 503 and "SCIF_LLM_SPEND_FILE" in r.json()["detail"]
    assert bedrock.calls == []


def test_seconds_to_midnight():
    from datetime import datetime, timezone
    assert llm.seconds_to_utc_midnight(datetime(2026, 11, 24, 23, 59, 0, tzinfo=timezone.utc)) == 60
