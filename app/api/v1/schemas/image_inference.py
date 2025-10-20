from typing import List, Optional
from pydantic import BaseModel

class InferenceResponse(BaseModel):
    label: str            # "Normal" | "Benign" | "Malignant"
    birads: str           # "BI-RADS 1/2/5"
    confidence: float     # 0..1
    filename: str

class InferenceTaggedResponse(InferenceResponse):
    tagged_filename: str

class ResultItem(BaseModel):
    id: int
    filename: str
    label: Optional[str] = None
    birads: Optional[str] = None
    confidence: Optional[float] = None
    tagged_filename: Optional[str] = None
    submitted_at: Optional[str] = None

class ResultsResponse(BaseModel):
    items: List[ResultItem]
    total: int