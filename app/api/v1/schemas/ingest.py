from typing import Optional, List
from pydantic import BaseModel

class PredResult(BaseModel):
    label: str
    birads: str
    confidence: float

class IngestItem(BaseModel):
    original_filename: str
    saved_as: str                    # chemin du fichier uploadé (raw)
    kind: str                        # "photo" | "dicom"
    anonymized_image_id: Optional[str] = None  # ex: "bfa001"
    anonymized_png: Optional[str] = None       # chemin vers dataset_bfa/images/bfa001.png
    anonymized_dicom: Optional[str] = None     # si DICOM anonymisé
    age_value: Optional[int] = None
    age_source: Optional[str] = None           # "ocr" | "imputed" | "none"

    prediction: Optional[PredResult] = None    # si run_inference=True

class IngestResponse(BaseModel):
    processed: List[IngestItem]
    counts: dict
