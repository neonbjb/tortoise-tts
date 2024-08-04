from pydantic import BaseModel

class TranscriptionRequest(BaseModel):
    text: str
    voice: str
    output_path: str
    preset: str = "ultra_fast"
