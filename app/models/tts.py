import os
from pydantic import BaseModel

from tortoise.api import MODELS_DIR

class TTSArgs(BaseModel):
    text: str
    voice: str = 'random'
    output_path: str = 'results/'
    preset: str = 'fast'
    model_dir: str = os.getenv("TORTOISE_MODELS_DIR", MODELS_DIR)
    use_deepspeed: bool = False
    kv_cache: bool = True
    half: bool = True
    candidates: int = 3
    seed: int = None
    cvvp_amount: float = 0.0
    produce_debug_state: bool = True
