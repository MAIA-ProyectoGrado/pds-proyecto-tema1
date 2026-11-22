from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.api import config
from src.api.model_loader import DEFAULT_MODEL

# ── clasificación ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    citation_context: str = Field(
        ...,
        max_length=config.MAX_CONTEXT_CHARS,
        description="Oración (o fragmento) que contiene la cita.",
        examples=["We fine-tune SciBERT [4] on our citation corpus and keep its in-domain vocabulary."],
    )
    rhetorical_section: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Sección del artículo citante donde aparece la cita. null si no se conoce.",
        examples=["Methods"],
    )
    model: str = Field(default=DEFAULT_MODEL, max_length=50, examples=["scibert"])
    #: artículo citado, opcional: id o URL de arXiv, o su texto (abstract o completo)
    arxiv_id: Optional[str] = Field(
        default=None, max_length=200,
        description="Id o URL de arXiv del artículo citado. SciBERT lee 3 oraciones clave de su abstract.",
        examples=["1903.10676"],
    )
    cited_text: Optional[str] = Field(
        default=None, max_length=config.MAX_CITED_TEXT_CHARS,
        description="Texto del artículo citado (abstract o completo), si no está en arXiv.",
    )


class KeySentence(BaseModel):
    rank: int
    text: str
    #: coseno entre la oración y el contexto de cita (MiniLM)
    similarity: float


class CitedPaper(BaseModel):
    id: Optional[str]
    title: Optional[str]


class KeySentencesInfo(BaseModel):
    #: used | not_provided | not_read_by_model | failed
    status: str
    #: arxiv-abstract | arxiv-title | text-abstract | text
    source: Optional[str]
    #: explicación para el usuario cuando no se usaron, o se usaron con una salvedad
    detail: Optional[str]
    cited_paper: Optional[CitedPaper]
    sentences: List[KeySentence]


class PredictLatency(BaseModel):
    cited_paper: float     # consulta a arXiv (0 si estaba en caché o no aplica)
    key_sentences: float   # selección de las oraciones clave con MiniLM
    inference: float       # clasificador
    total: float


class PredictResponse(BaseModel):
    predicted_label: str
    predicted_label_display: str
    confidence: float
    probabilities: Dict[str, float]
    ambiguous: bool
    model_id: str
    model_version: str
    #: sección tal como se recibió (null si no se indicó)
    rhetorical_section: Optional[str]
    #: el modelo leyó oraciones clave del artículo citado
    passages_used: bool
    key_sentences: KeySentencesInfo
    #: entrada efectiva del modelo, para auditar qué se clasificó realmente. En
    #: SciBERT es el par "contexto [SEP] Section: ... | Cited passages: ..."
    input_text: str
    latency_ms: PredictLatency
    status: str


# ── catálogo y estado ────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    id: str
    name: str
    kind: str
    #: qué lee el modelo, para explicarlo en el selector
    reads: str
    reads_passages: bool
    version: str
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
    #: las oraciones clave requieren el codificador MiniLM
    key_sentences_available: bool


class ComponentStatus(BaseModel):
    available: bool
    loaded: bool


class HealthResponse(BaseModel):
    #: ok: el modelo por defecto y lo que necesita están en disco; degraded: falta algo
    status: str
    version: str
    uptime_s: float
    default_model: str
    models: Dict[str, ComponentStatus]
    key_sentences_encoder: ComponentStatus
    retrieval_available: bool
    #: parámetros efectivos (variables SCIF_*)
    config: Dict[str, object]


# ── recuperación local de pasajes ───────────────────────────────────────────

class RetrieveRequest(BaseModel):
    citation_context: str = Field(
        ...,
        max_length=config.MAX_CONTEXT_CHARS,
        examples=["We build our citation encoder on top of SciBERT [4] and keep its in-domain vocabulary unchanged."],
    )
    #: uno de los dos: identificador (o URL) de arXiv, o el texto completo del artículo citado
    arxiv_id: Optional[str] = Field(default=None, max_length=200, examples=["1903.10676"])
    cited_text: Optional[str] = Field(default=None, max_length=config.MAX_CITED_TEXT_CHARS)
    top_k: int = Field(default=config.TOP_K, ge=1, le=10)


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
