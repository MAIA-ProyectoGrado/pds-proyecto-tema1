from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.api.model_loader import DEFAULT_MODEL


class PredictRequest(BaseModel):
    citation_context: str = Field(
        ...,
        examples=["Although several domain-tailored approaches have been proposed, it is still an open problem."],
    )
    rhetorical_section: Optional[str] = Field(default="Abstract", examples=["Introduction"])
    model: str = Field(default=DEFAULT_MODEL, examples=["scibert"])


class PredictResponse(BaseModel):
    predicted_label: str
    predicted_label_display: str
    confidence: float
    probabilities: Dict[str, float]
    ambiguous: bool
    model_id: str
    #: entrada efectiva enviada al modelo (`contexto [SEC] sección canónica`),
    #: expuesta para que el tablero pueda auditar qué se clasificó realmente.
    input_text: str
    latency_ms: float
    status: str


class ModelInfo(BaseModel):
    id: str
    name: str
    kind: str
    available: bool
    loaded: bool
    recommended: bool


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    default_model: str
    labels: List[str]
    label_display: Dict[str, str]
    sections: List[str]
    #: la recuperación de pasajes requiere el encoder SciBERT
    retrieval_available: bool


# ── recuperación local de pasajes ───────────────────────────────────────────

class RetrieveRequest(BaseModel):
    citation_context: str = Field(
        ...,
        examples=["We build our citation encoder on top of SciBERT [4] and keep its in-domain vocabulary unchanged."],
    )
    #: uno de los dos: identificador (o URL) de arXiv, o el texto completo del artículo citado
    arxiv_id: Optional[str] = Field(default=None, examples=["1903.10676"])
    cited_text: Optional[str] = Field(default=None)
    top_k: int = Field(default=3, ge=1, le=10)


class PaperInfo(BaseModel):
    id: Optional[str]
    title: Optional[str]
    authors: List[str]
    year: str
    source: str            # "arxiv-pdf" | "text"


class RetrievedChunk(BaseModel):
    rank: int
    index: int             # posición del chunk dentro del documento (0-based)
    section: str           # sección retórica canónica de origen
    text: str
    n_words: int
    cosine: float


class RetrieveLatency(BaseModel):
    download: float
    embed_document: float
    total: float


class RetrieveResponse(BaseModel):
    paper: PaperInfo
    n_chunks: int
    n_words: int
    chunks: List[RetrievedChunk]
    #: similitud de cada chunk del documento, en orden; alimenta el mapa del documento
    scores: List[float]
    sections: List[str]
    cached: bool
    latency_ms: RetrieveLatency
    status: str
