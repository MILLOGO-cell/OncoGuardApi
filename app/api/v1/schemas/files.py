from pydantic import BaseModel
from typing import Optional

class FileItem(BaseModel):
    kind: str                 # "png" | "dicom"
    filename: str             # ex: bfa001.png
    size_bytes: int
    created_at: float         # timestamp epoch (secondes)
    download_url: str         # endpoint pour télécharger
