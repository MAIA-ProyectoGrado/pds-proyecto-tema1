"""Selección de las oraciones clave del artículo citado (entrada del modelo v4).

Replica la sección 7 del notebook `Obtener_dataset.ipynb`: el texto del artículo
citado (su abstract o, en su defecto, su título) se divide en oraciones y se
eligen las 3 más parecidas al contexto de cita según `all-MiniLM-L6-v2`.

El cálculo de los vectores reproduce a `sentence-transformers` (promedio de los
tokens ponderado por la máscara de atención y normalización L2, con un máximo de
256 tokens), sin depender de esa librería.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256
MIN_SENTENCE_CHARS = 13          # el notebook descarta oraciones con 12 caracteres o menos


@dataclass
class Passage:
    rank: int
    text: str
    similarity: float
    position_char: int


def split_sentences(text: str | None) -> list[str]:
    """Segmentador del notebook: corrige 'palabra.Mayúscula', parte en . ! ? y quita duplicados."""
    if text is None or not str(text).strip() or str(text).strip().lower() == "nan":
        return []
    norm = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", str(text).strip())
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", norm) if len(s.strip()) >= MIN_SENTENCE_CHARS]
    seen, unique = set(), []
    for s in sents:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique.append(s)
    return unique


class SentenceEncoder:
    """MiniLM con el pooling de sentence-transformers. Carga diferida y segura entre hilos."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def available(self) -> bool:
        return (self.model_dir / "config.json").exists()

    def _load(self):
        with self._lock:
            if self._loaded:
                return
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))
            self._model = AutoModel.from_pretrained(str(self.model_dir)).eval()
            cfg = self.model_dir / "pooling.json"
            self._max_len = json.loads(cfg.read_text()).get("max_seq_length", MAX_SEQ_LENGTH) if cfg.exists() else MAX_SEQ_LENGTH
            self._loaded = True

    def encode(self, texts: list[str], batch: int = 32):
        self._load()
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self._tok(texts[i:i + batch], padding=True, truncation=True,
                                max_length=self._max_len, return_tensors="pt")
                h = self._model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(v, p=2, dim=-1))
        return torch.cat(out)


def select_passages(citation_context: str, source_text: str | None, encoder: SentenceEncoder,
                    k: int = 3, max_candidates: int | None = None) -> list[Passage]:
    """Las k oraciones de `source_text` más parecidas al contexto, como en el notebook.

    `max_candidates` limita cuántas oraciones se comparan (las primeras del texto).
    El entrenamiento no lo usa, porque sus fuentes son abstracts; la API lo aplica
    cuando el texto pegado no tiene un abstract reconocible y puede ser largo.
    """
    ctx = str(citation_context or "").strip()
    src = str(source_text or "").strip()
    if not ctx or not src or src.lower() == "nan":
        return []
    candidates = split_sentences(src) or [src]
    if max_candidates:
        candidates = candidates[:max_candidates]
    vecs = encoder.encode([ctx] + candidates)
    sims = (vecs[1:] @ vecs[0]).tolist()
    order = sorted(range(len(candidates)), key=lambda i: -sims[i])[:k]
    passages = []
    for rank, i in enumerate(order, start=1):
        text = candidates[i]
        if not text.endswith((".", "!", "?")):
            text += "."
        passages.append(Passage(rank, text, round(float(sims[i]), 4), max(0, src.find(candidates[i]))))
    return passages
