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
    f1_macro_val: Optional[float]
    metric_verified: bool
    metric_note: str


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    default_model: str
    labels: List[str]
    label_display: Dict[str, str]
    sections: List[str]
