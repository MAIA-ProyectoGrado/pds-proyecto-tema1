"""Recuperación local de pasajes: Top-k chunks del artículo citado.

Dado un contexto de cita y el artículo citado (por id de arXiv o texto pegado),
segmenta el documento en chunks de hasta 300 palabras respetando oraciones y
secciones, los representa con el mismo encoder SciBERT que sirve el clasificador
y devuelve los k más similares por coseno.

La técnica (SciBERT + segmentación oracional + coseno) es la del pipeline de
`notebooks/procesamiento.ipynb`; la diferencia es que aquí se aplica al texto del
documento citado y no al propio contexto de cita.
"""
from __future__ import annotations

import io
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import certifi
import pdfplumber
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.scif_predict import canon_section  # noqa: E402
from src.api import model_loader  # noqa: E402

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{id}"
ATOM = {"a": "http://www.w3.org/2005/Atom"}

MAX_WORDS_PER_CHUNK = 300
MIN_WORDS_DOCUMENT = 50
MAX_LEN_TOKENS = 256
MIN_WORDS_PER_CHUNK = 20      # descarta residuos: marca de agua de arXiv, pies de figura sueltos
X_TOLERANCE = 2               # pdfplumber pega palabras con el valor por defecto (3)
HTTP_TIMEOUT = 40
USER_AGENT = "scif-citation-dashboard/1.0 (academic project)"


# ── errores ─────────────────────────────────────────────────────────────────

class InvalidArxivId(ValueError):
    """La cadena no tiene forma de identificador de arXiv."""


class PaperNotFound(LookupError):
    """arXiv no tiene ningún artículo con ese identificador."""


class DownloadError(RuntimeError):
    """No se pudo descargar o leer el PDF."""


class TextTooShort(ValueError):
    """El documento no tiene texto suficiente para segmentar."""


# ── identificadores de arXiv ────────────────────────────────────────────────

_NEW_ID = r"\d{4}\.\d{4,5}(?:v\d+)?"
_OLD_ID = r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"
_ID_RE = re.compile(rf"(?:arxiv\.org/(?:abs|pdf)/)?({_NEW_ID}|{_OLD_ID})(?:\.pdf)?/?$", re.I)


def parse_arxiv_id(raw: str) -> str:
    """Acepta `1903.10676`, `1903.10676v2`, `arXiv:1903.10676` o la URL completa."""
    s = (raw or "").strip()
    s = re.sub(r"^(https?://)?(www\.)?", "", s, flags=re.I)
    s = re.sub(r"^arxiv:\s*", "", s, flags=re.I)
    m = _ID_RE.search(s)
    if not m:
        raise InvalidArxivId(raw)
    return m.group(1)


# ── descarga ────────────────────────────────────────────────────────────────

def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    s.verify = certifi.where()
    return s


def fetch_arxiv_metadata(arxiv_id: str) -> dict[str, Any]:
    r = _session().get(ARXIV_API, params={"id_list": arxiv_id}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    entry = root.find("a:entry", ATOM)
    # arXiv responde con una entrada de error, no con 404, cuando el id no existe
    if entry is None or entry.find("a:title", ATOM) is None or (entry.find("a:title", ATOM).text or "").strip() == "Error":
        raise PaperNotFound(arxiv_id)
    title = re.sub(r"\s+", " ", entry.find("a:title", ATOM).text or "").strip()
    authors = [
        (a.find("a:name", ATOM).text or "").strip()
        for a in entry.findall("a:author", ATOM)
        if a.find("a:name", ATOM) is not None
    ]
    published = (entry.findtext("a:published", default="", namespaces=ATOM) or "")[:4]
    return {"id": arxiv_id, "title": title, "authors": authors, "year": published, "source": "arxiv-pdf"}


def fetch_arxiv_pdf_text(arxiv_id: str) -> str:
    try:
        r = _session().get(ARXIV_PDF.format(id=arxiv_id), timeout=HTTP_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise DownloadError(f"No se pudo descargar el PDF de arXiv:{arxiv_id}: {exc}") from exc
    if not r.content.startswith(b"%PDF"):
        raise DownloadError(f"arXiv:{arxiv_id} no devolvió un PDF (¿artículo retirado o solo fuente LaTeX?).")
    return pdf_to_text(r.content)


# ── extracción de texto ─────────────────────────────────────────────────────

def _page_text(page) -> str:
    """Texto de una página. Si la página está a dos columnas (casi ninguna palabra
    cruza el eje central), se extraen las columnas por separado para no entreverar
    las líneas — el defecto habitual de pdfplumber con los PDF de arXiv."""
    words = page.extract_words()
    if not words:
        return ""
    mid = page.width / 2
    crossing = sum(1 for w in words if w["x0"] < mid - 6 and w["x1"] > mid + 6)
    if len(words) > 60 and crossing / len(words) < 0.06:
        left = page.crop((0, 0, mid, page.height)).extract_text(x_tolerance=X_TOLERANCE) or ""
        right = page.crop((mid, 0, page.width, page.height)).extract_text(x_tolerance=X_TOLERANCE) or ""
        return left + "\n" + right
    return page.extract_text(x_tolerance=X_TOLERANCE) or ""


def pdf_to_text(data: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [_page_text(p) for p in pdf.pages]
    except Exception as exc:  # pdf corrupto, cifrado, etc.
        raise DownloadError(f"No se pudo leer el PDF: {exc}") from exc
    text = "\n".join(pages)
    text = re.sub(r"-\n(?=[a-z])", "", text)   # une palabras partidas por guion
    return text


# ── secciones y chunks ──────────────────────────────────────────────────────

_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|background|related work|preliminaries|"
    r"methods?|methodology|approach|model|experiments?|experimental setup|"
    r"results|evaluation|discussion|conclusions?|limitations|"
    r"references|bibliography|acknowledge?ments?|appendix)\b[\s:.]*$",
    re.I,
)
_STOP_SECTIONS = {"references", "bibliography", "acknowledgements", "acknowledgments", "appendix"}
_STOP_AT_LINE_START = re.compile(r"^\s*(?:\d+\.?\s+)?(references|bibliography|acknowledge?ments?|appendix)\b", re.I)


@dataclass
class Chunk:
    index: int
    section: str
    text: str
    n_words: int


def split_sections(text: str) -> list[tuple[str, str]]:
    """Divide el texto por encabezados reconocibles. Devuelve [(sección, texto)].
    Lo que sigue a References/Appendix se descarta."""
    sections: list[tuple[str, list[str]]] = [("Abstract", [])]
    for line in text.split("\n"):
        stripped = line.strip()
        if _STOP_AT_LINE_START.match(stripped):
            break
        m = _HEADING_RE.match(stripped) if 0 < len(stripped.split()) <= 6 else None
        if m:
            name = m.group(1).lower()
            if name in _STOP_SECTIONS:
                break
            sections.append((canon_section(name), []))
            continue
        if stripped:
            sections[-1][1].append(stripped)
    out = [(name, " ".join(lines)) for name, lines in sections if lines]
    return out or [("Desconocida", re.sub(r"\s+", " ", text))]


_SENT_SPLIT = re.compile(r"(?<!\bet al\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bFig\.)(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(re.sub(r"\s+", " ", text)) if len(s.strip()) > 3]


def pack_chunks(sentences: list[str], section: str, start_index: int,
                max_words: int = MAX_WORDS_PER_CHUNK) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[str] = []
    n = 0
    for s in sentences:
        w = len(s.split())
        if cur and n + w > max_words:
            chunks.append(Chunk(start_index + len(chunks), section, " ".join(cur), n))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(Chunk(start_index + len(chunks), section, " ".join(cur), n))
    return chunks


def build_chunks(text: str) -> list[Chunk]:
    raw: list[Chunk] = []
    for section, body in split_sections(text):
        raw.extend(pack_chunks(split_sentences(body), section, len(raw)))
    kept = [c for c in raw if c.n_words >= MIN_WORDS_PER_CHUNK] or raw
    return [Chunk(i, c.section, c.text, c.n_words) for i, c in enumerate(kept)]


# ── embeddings ──────────────────────────────────────────────────────────────

def _encoder():
    """Encoder SciBERT y tokenizador del clasificador ya cargado. Se reutiliza el
    mismo artefacto en memoria; no se carga una segunda copia."""
    p = model_loader.get_predictor("scibert")
    return p._tok, p._model.base_model, p._torch


def embed(texts: list[str], batch: int = 8):
    tok, enc, torch = _encoder()
    outs = []
    with model_loader._infer_lock, torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True,
                    max_length=MAX_LEN_TOKENS, return_tensors="pt")
            h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            outs.append((h * m).sum(1) / m.sum(1).clamp(min=1))   # mean pooling
    v = torch.cat(outs)
    return torch.nn.functional.normalize(v, p=2, dim=-1)


def retrieval_available() -> bool:
    return model_loader.is_available("scibert")


# ── caché de documentos ─────────────────────────────────────────────────────

_doc_cache: dict[str, dict[str, Any]] = {}
_doc_lock = threading.Lock()


def _prepare_document(text: str, meta: dict[str, Any]) -> dict[str, Any]:
    n_words = len(text.split())
    if n_words < MIN_WORDS_DOCUMENT:
        raise TextTooShort(f"El documento tiene {n_words} palabras; se necesitan al menos {MIN_WORDS_DOCUMENT}.")
    chunks = build_chunks(text)
    t = time.perf_counter()
    vectors = embed([c.text for c in chunks])
    return {
        "meta": meta, "n_words": n_words, "chunks": chunks, "vectors": vectors,
        "embed_ms": round((time.perf_counter() - t) * 1000, 1),
    }


def load_arxiv_document(raw_id: str) -> dict[str, Any]:
    arxiv_id = parse_arxiv_id(raw_id)
    with _doc_lock:
        if arxiv_id in _doc_cache:
            doc = dict(_doc_cache[arxiv_id])
            doc["download_ms"] = 0.0
            doc["cached"] = True
            return doc
    t = time.perf_counter()
    meta = fetch_arxiv_metadata(arxiv_id)
    text = fetch_arxiv_pdf_text(arxiv_id)
    download_ms = round((time.perf_counter() - t) * 1000, 1)
    doc = _prepare_document(text, meta)
    doc["download_ms"] = download_ms
    doc["cached"] = False
    with _doc_lock:
        _doc_cache[arxiv_id] = doc
    return doc


def load_text_document(text: str) -> dict[str, Any]:
    meta = {"id": None, "title": None, "authors": [], "year": "", "source": "text"}
    doc = _prepare_document(text, meta)
    doc["download_ms"] = 0.0
    doc["cached"] = False
    return doc


# ── recuperación ────────────────────────────────────────────────────────────

def retrieve(citation_context: str, *, arxiv_id: str | None = None,
             cited_text: str | None = None, top_k: int = 3) -> dict[str, Any]:
    t0 = time.perf_counter()
    if arxiv_id:
        doc = load_arxiv_document(arxiv_id)
    elif cited_text and cited_text.strip():
        doc = load_text_document(cited_text)
    else:
        raise ValueError("Se requiere arxiv_id o cited_text.")

    chunks: list[Chunk] = doc["chunks"]
    q = embed([re.sub(r"\s+", " ", citation_context).strip()])
    sims = (doc["vectors"] @ q.T).squeeze(1)
    k = max(1, min(top_k, len(chunks)))
    top = sims.topk(k)

    return {
        "paper": doc["meta"],
        "n_chunks": len(chunks),
        "n_words": doc["n_words"],
        "chunks": [
            {"rank": r + 1, **asdict(chunks[int(i)]), "cosine": round(float(s), 4)}
            for r, (s, i) in enumerate(zip(top.values, top.indices))
        ],
        "scores": [round(float(s), 4) for s in sims],
        "sections": [c.section for c in chunks],
        "cached": doc["cached"],
        "latency_ms": {
            "download": doc["download_ms"],
            "embed_document": doc["embed_ms"] if not doc["cached"] else 0.0,
            "total": round((time.perf_counter() - t0) * 1000, 1),
        },
    }
