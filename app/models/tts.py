import os

from pydantic import BaseModel

class TTSArgs(BaseModel):
    text: str
    voice: str
    output_path: str
    preset: str
    model_dir: str = os.getenv("TORTOISE_MODELS_DIR", "data/models")
    use_deepspeed: bool = False
    kv_cache: bool = False
    half: bool = False
    candidates: int = 1
    seed: int = None
    cvvp_amount: float = 0.0
    produce_debug_state: bool = False
