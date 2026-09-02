from pydantic import BaseModel, Field
from typing import Dict, Optional

class PredictRequest(BaseModel):
    citation_context: str = Field(
        ..., 
        example="Although several domain-tailored approaches have been proposed, it is still an open problem."
    )
    rhetorical_section: Optional[str] = Field(
        default="Abstract", 
        example="Abstract"
    )

class PredictResponse(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    status: str