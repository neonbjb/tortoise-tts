from enum import Enum
from typing import Optional
from pydantic import BaseModel

class Presets(str, Enum):
    ULTRA_FAST='ultra_fast'
    FAST='fast'
    STANDARD='standard'
    HIGH_QUALITY='high_quality'
    
class TranscriptionRequest(BaseModel):
    text: str
    voice: str
    preset: Presets = "ultra_fast"
