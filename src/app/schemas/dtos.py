from typing import Dict, Optional

from pydantic import BaseModel


class AnalysisResult(BaseModel):
    filename: str
    prediction: str
    probability_real: float
    probability_fake: float
    features: Dict[str, float]
    processing_time_ms: Optional[float] = None
