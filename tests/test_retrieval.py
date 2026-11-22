"""Pruebas de la recuperación local de pasajes (Top-k chunks).

Sin red: el documento de prueba es sintético y la descarga de arXiv se sustituye
con monkeypatch. Los casos que necesitan el encoder SciBERT se omiten si faltan
torch o los pesos, igual que en test_api.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from src.api import model_loader, retrieval as R  # noqa: E402
from src.api.main import app  # noqa: E402

scibert_required = pytest.mark.skipif(
    not model_loader.is_available("scibert"), reason="faltan los pesos de SciBERT (DVC)"
)

# ── documento sintético con encabezados y un pasaje objetivo inconfundible ──

TARGET = ("The vocabulary is built with SentencePiece on the scientific corpus and the "
          "resulting subword units are kept unchanged during fine-tuning.")

DOC = "\n".join([
    "A Synthetic Paper About Encoders",
    "Abstract",
    " ".join(f"This abstract sentence number {i} describes the general goal of the work." for i in range(6)),
    "1 Introduction",
    " ".join(f"Introduction sentence {i} motivates the problem of citation analysis." for i in range(20)),
    "2 Methods",
    " ".join(f"Method sentence {i} explains a training detail of the encoder." for i in range(8)),
    TARGET,
    " ".join(f"Method sentence {i} explains another training detail." for i in range(8)),
    "3 Experiments",
    " ".join(f"Experiment sentence {i} reports a number on a benchmark." for i in range(20)),
    "4 Conclusion",
    "We conclude that the approach works. Future work will extend it.",
    "Acknowledgments We thank the reviewers for their comments.",
    "References",
    "Devlin et al. 2019. BERT. Beltagy et al. 2019. SciBERT.",
])


@pytest.fixture(scope="module")
def client():
    with fastapi_testclient.TestClient(app) as c:
        yield c


# ── identificadores ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("1903.10676", "1903.10676"),
    ("1903.10676v2", "1903.10676v2"),
    ("arXiv:1903.10676", "1903.10676"),
    ("https://arxiv.org/abs/1903.10676", "1903.10676"),
    ("https://arxiv.org/pdf/1903.10676.pdf", "1903.10676"),
    ("cs/0007010v1", "cs/0007010v1"),
])
def test_parse_arxiv_id(raw, expected):
    assert R.parse_arxiv_id(raw) == expected


@pytest.mark.parametrize("raw", ["", "hola", "10.1000/xyz", "1903"])
def test_parse_arxiv_id_rejects_garbage(raw):
    with pytest.raises(R.InvalidArxivId):
        R.parse_arxiv_id(raw)


# ── secciones y chunks (puro texto, sin modelo) ─────────────────────────────

def test_sections_are_detected_and_references_dropped():
    secs = dict(R.split_sections(DOC))
    assert {"Abstract", "Introduccion", "Metodos", "Experimentos", "Conclusion"} <= set(secs)
    joined = " ".join(secs.values())
    assert "Devlin et al. 2019. BERT" not in joined      # lo posterior a References se descarta
    assert "We thank the reviewers" not in joined         # Acknowledgments pegado al cuerpo también


def test_chunks_respect_limit_and_sentence_boundaries():
    chunks = R.build_chunks(DOC)
    assert chunks, "sin chunks"
    for c in chunks:
        assert c.n_words <= R.MAX_WORDS_PER_CHUNK
        assert c.n_words >= R.MIN_WORDS_PER_CHUNK
        assert c.text.rstrip()[-1] in ".!?"              # no corta a mitad de oración
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_target_sentence_lands_in_a_methods_chunk():
    chunks = R.build_chunks(DOC)
    holder = [c for c in chunks if TARGET in c.text]
    assert len(holder) == 1
    assert holder[0].section == "Metodos"


def test_sentence_split_keeps_abbreviations():
    s = R.split_sentences("As shown by Devlin et al. (2019), e.g. in Fig. 2, it works. Next sentence.")
    assert s == ["As shown by Devlin et al. (2019), e.g. in Fig. 2, it works.", "Next sentence."]


def test_text_too_short_is_rejected():
    with pytest.raises(R.TextTooShort):
        R.load_text_document("solo unas pocas palabras aquí")


def test_abstract_is_recognized_in_pasted_full_text():
    abstract = R.abstract_of(DOC)
    assert abstract.startswith("This abstract sentence number 0")
    assert "Introduction sentence" not in abstract


def test_pasted_text_without_headings_has_no_abstract():
    assert R.abstract_of("Only an abstract pasted as plain text. " * 10) is None


# ── arXiv: errores de red, caché de metadatos y tope del PDF (sin red) ──────

class _FakeResponse:
    def __init__(self, content=b"", headers=None, text=""):
        self.content, self.headers, self.text = content, headers or {}, text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]


class _FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error

    def get(self, *_a, **_k):
        if self.error:
            raise self.error
        return self.response


def test_arxiv_network_error_becomes_download_error(monkeypatch):
    import requests
    monkeypatch.setattr(R, "_session", lambda: _FakeSession(error=requests.ConnectionError("sin red")))
    with pytest.raises(R.DownloadError, match="No se pudo consultar arXiv"):
        R.fetch_arxiv_metadata("1903.10676")


ATOM_ENTRY = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<title>SciBERT: A Pretrained
  Language Model</title><summary>  We release SciBERT.
  It is trained on papers. </summary><published>2019-03-26T00:00:00Z</published>
<author><name>Iz Beltagy</name></author></entry></feed>"""


def test_arxiv_metadata_includes_normalized_abstract(monkeypatch):
    monkeypatch.setattr(R, "_session", lambda: _FakeSession(_FakeResponse(text=ATOM_ENTRY)))
    meta = R.fetch_arxiv_metadata("1903.10676")
    assert meta["title"] == "SciBERT: A Pretrained Language Model"
    assert meta["abstract"] == "We release SciBERT. It is trained on papers."
    assert meta["authors"] == ["Iz Beltagy"] and meta["year"] == "2019"


def test_metadata_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(R, "fetch_arxiv_metadata", lambda i: calls.append(i) or {"id": i, "title": "T"})
    R._meta_cache.pop("2401.00003", None)
    assert R.get_arxiv_metadata("2401.00003") is R.get_arxiv_metadata("2401.00003")
    assert calls == ["2401.00003"]
    R._meta_cache.pop("2401.00003", None)


def test_pdf_larger_than_limit_is_rejected(monkeypatch):
    monkeypatch.setattr(R, "MAX_PDF_BYTES", 1000)
    big = _FakeResponse(content=b"%PDF" + b"0" * 5000)
    monkeypatch.setattr(R, "_session", lambda: _FakeSession(big))
    with pytest.raises(R.DownloadError, match="supera el máximo"):
        R.fetch_arxiv_pdf_text("1903.10676")


def test_retrieve_maps_arxiv_outage_to_502(client, monkeypatch):
    def outage(_id):
        raise R.DownloadError("No se pudo consultar arXiv: timeout.")

    monkeypatch.setattr(R, "fetch_arxiv_metadata", outage)
    R._doc_cache.pop("2401.00004", None)
    r = client.post("/retrieve", json={"citation_context": "x", "arxiv_id": "2401.00004"})
    assert r.status_code == 502


# ── endpoint: validaciones que no necesitan modelo ──────────────────────────

def test_retrieve_requires_context(client):
    assert client.post("/retrieve", json={"citation_context": "  ", "arxiv_id": "1903.10676"}).status_code == 400


def test_retrieve_requires_a_document(client):
    assert client.post("/retrieve", json={"citation_context": "x"}).status_code == 400


def test_retrieve_rejects_invalid_arxiv_id(client):
    assert client.post("/retrieve", json={"citation_context": "x", "arxiv_id": "hola"}).status_code == 422


def test_models_reports_retrieval_availability(client):
    body = client.get("/models").json()
    assert isinstance(body["retrieval_available"], bool)


# ── ranking real con SciBERT (opcional) ─────────────────────────────────────

@scibert_required
def test_target_chunk_is_ranked_first(client):
    pytest.importorskip("torch")
    ctx = ("We keep SciBERT's in-domain vocabulary unchanged so that scientific "
           "subword units are preserved throughout fine-tuning.")
    r = client.post("/retrieve", json={"citation_context": ctx, "cited_text": DOC})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["paper"]["source"] == "text"
    assert body["n_chunks"] == len(body["scores"]) == len(body["sections"])
    assert len(body["chunks"]) == 3
    assert TARGET in body["chunks"][0]["text"]
    assert body["chunks"][0]["section"] == "Metodos"
    assert body["chunks"][0]["cosine"] == max(body["scores"])


@scibert_required
def test_arxiv_path_uses_cache_and_mocked_download(client, monkeypatch):
    pytest.importorskip("torch")
    calls = {"meta": 0, "pdf": 0}

    def fake_meta(arxiv_id):
        calls["meta"] += 1
        return {"id": arxiv_id, "title": "Fake Paper", "authors": ["A. Autor"], "year": "2024", "source": "arxiv-pdf"}

    def fake_pdf(arxiv_id):
        calls["pdf"] += 1
        return DOC

    monkeypatch.setattr(R, "fetch_arxiv_metadata", fake_meta)
    monkeypatch.setattr(R, "fetch_arxiv_pdf_text", fake_pdf)
    R._doc_cache.pop("2401.00001", None)
    R._meta_cache.pop("2401.00001", None)

    req = {"citation_context": "vocabulary and subword units during fine-tuning", "arxiv_id": "2401.00001"}
    first = client.post("/retrieve", json=req).json()
    second = client.post("/retrieve", json=req).json()

    assert first["paper"]["title"] == "Fake Paper"
    assert first["cached"] is False and second["cached"] is True
    assert calls == {"meta": 1, "pdf": 1}                # la segunda llamada no vuelve a descargar
    assert second["latency_ms"]["download"] == 0.0
    assert [c["index"] for c in first["chunks"]] == [c["index"] for c in second["chunks"]]


@scibert_required
def test_paper_not_found_is_404(client, monkeypatch):
    pytest.importorskip("torch")

    def boom(arxiv_id):
        raise R.PaperNotFound(arxiv_id)

    monkeypatch.setattr(R, "fetch_arxiv_metadata", boom)
    r = client.post("/retrieve", json={"citation_context": "x", "arxiv_id": "9999.99999"})
    assert r.status_code == 404
