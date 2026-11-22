"""Oraciones clave del artículo citado: la parte de la entrada de SciBERT
configuración B que viene del documento.

Se construyen como en el entrenamiento (`models/passages.py`): las 3 oraciones del
abstract del artículo citado más parecidas al contexto según MiniLM. El abstract
sale de arXiv o del texto pegado por el usuario.

La clasificación nunca falla por el artículo citado. Si no se indica, si arXiv no
responde o si falta el codificador, el modelo clasifica sin oraciones clave (como
en el entrenamiento, con "Passages unavailable") y la respuesta dice por qué.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.passages import Passage, select_passages, split_sentences  # noqa: E402
from src.api import model_loader, retrieval  # noqa: E402

K = 3                        # fijo: es el número de oraciones con que se entrenó
MAX_CANDIDATES = 60          # solo para texto pegado sin abstract reconocible

# Estados posibles (el tablero muestra un aviso distinto para cada uno)
USED = "used"                          # el modelo leyó las oraciones clave
NOT_PROVIDED = "not_provided"          # no se indicó artículo citado
NOT_READ_BY_MODEL = "not_read_by_model"  # el modelo elegido no lee el artículo
FAILED = "failed"                      # se indicó, pero no se pudo usar


@dataclass
class KeySentences:
    status: str
    source: str | None = None          # arxiv-abstract | arxiv-title | text-abstract | text
    detail: str | None = None
    paper_id: str | None = None
    title: str | None = None
    sentences: list[Passage] = field(default_factory=list)
    fetch_ms: float = 0.0
    encode_ms: float = 0.0

    @property
    def texts(self) -> list[str]:
        return [p.text for p in self.sentences]


def build(citation_context: str, *, reads_passages: bool,
          arxiv_id: str | None = None, cited_text: str | None = None) -> KeySentences:
    has_id = bool(arxiv_id and arxiv_id.strip())
    has_text = bool(cited_text and cited_text.strip())

    if not reads_passages:
        return KeySentences(NOT_READ_BY_MODEL,
                            detail="Este modelo solo lee el contexto y la sección; no usa el artículo citado.")
    if not has_id and not has_text:
        return KeySentences(NOT_PROVIDED,
                            detail="No se indicó el artículo citado: el modelo clasificó solo con el contexto y la sección.")
    if not model_loader.encoder_available():
        return KeySentences(FAILED, detail=(
            "Falta el codificador de oraciones clave (models/minilm-l6-v2). "
            "El modelo clasificó solo con el contexto y la sección."))

    paper_id = title = None
    t = time.perf_counter()
    if has_id:
        try:
            paper_id = retrieval.parse_arxiv_id(arxiv_id)
            meta = retrieval.get_arxiv_metadata(paper_id)
        except retrieval.InvalidArxivId:
            return KeySentences(FAILED, detail=f"'{arxiv_id.strip()}' no es un identificador de arXiv válido. "
                                               "El modelo clasificó sin el artículo citado.")
        except retrieval.PaperNotFound:
            return KeySentences(FAILED, paper_id=paper_id,
                                detail=f"arXiv no tiene ningún artículo con id '{paper_id}'. "
                                       "El modelo clasificó sin el artículo citado.",
                                fetch_ms=_ms(t))
        except retrieval.DownloadError as exc:
            return KeySentences(FAILED, paper_id=paper_id,
                                detail=f"{exc} El modelo clasificó sin el artículo citado.",
                                fetch_ms=_ms(t))
        title = meta.get("title") or None
        if meta.get("abstract"):
            source, text, cap = "arxiv-abstract", meta["abstract"], None
        elif title:
            source, text, cap = "arxiv-title", title, None     # misma prioridad que el entrenamiento
        else:
            return KeySentences(FAILED, paper_id=paper_id, fetch_ms=_ms(t),
                                detail="arXiv no devolvió abstract ni título. El modelo clasificó sin el artículo citado.")
    else:
        abstract = retrieval.abstract_of(cited_text)
        if abstract:
            source, text, cap = "text-abstract", abstract, None
        else:
            source, text, cap = "text", cited_text, MAX_CANDIDATES
    fetch_ms = _ms(t)

    t = time.perf_counter()
    with model_loader._infer_lock:
        sentences = select_passages(citation_context, text, model_loader.encoder(), k=K, max_candidates=cap)
    encode_ms = _ms(t)
    if not sentences:
        return KeySentences(FAILED, source=source, paper_id=paper_id, title=title,
                            detail="El texto del artículo citado no tiene oraciones utilizables. "
                                   "El modelo clasificó sin el artículo citado.",
                            fetch_ms=fetch_ms, encode_ms=encode_ms)
    detail = None
    if source == "arxiv-title":
        detail = "arXiv no tiene el abstract de este artículo; el modelo leyó su título."
    elif source == "text" and len(split_sentences(text)) > MAX_CANDIDATES:
        detail = (f"No se reconoció un abstract en el texto pegado; las oraciones clave salen de sus "
                  f"primeras {MAX_CANDIDATES} oraciones.")
    return KeySentences(USED, source=source, detail=detail, paper_id=paper_id, title=title,
                        sentences=sentences, fetch_ms=fetch_ms, encode_ms=encode_ms)


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 1)
