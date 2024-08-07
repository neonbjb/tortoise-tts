import os
from pydantic import BaseModel
from enum import Enum

from tortoise.do_tts import pick_best_batch_size_for_gpu
from tortoise.api import MODELS_DIR

from .request import Presets

class TTSArgs(BaseModel):
    text: str
    voice: str = 'random'
    output_path: str = 'data/samples/'
    preset: Presets = 'ultra_fast'
    model_dir: str = os.getenv("TORTOISE_MODELS_DIR", MODELS_DIR)
    use_deepspeed: bool = False
    kv_cache: bool = True
    autoregressive_batch_size: int = pick_best_batch_size_for_gpu()
    half: bool = True
    candidates: int = 1
    seed: int = None
    cvvp_amount: float = 0.0
    produce_debug_state: bool = True
