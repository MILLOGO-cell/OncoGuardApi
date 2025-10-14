from pydantic import BaseModel

class InferenceResponse(BaseModel):
    label: str            # "Normal" | "Benign" | "Malignant"
    birads: str           # "BI-RADS 1/2/5"
    confidence: float     # 0..1
    filename: str
