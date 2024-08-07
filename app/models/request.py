from typing import Optional
from pydantic import BaseModel

class TranscriptionRequest(BaseModel):
    text: str
    voice: str
    output_path: Optional[str] = "data/samples/"
    preset: str = "ultra_fast"
