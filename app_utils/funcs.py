import gc
import os
from contextlib import contextmanager
from time import time
from typing import Optional

import streamlit as st

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voices


@contextmanager
def timeit(desc=""):
    start = time()
    yield
    print(f"{desc} took {time() - start:.2f} seconds")


@st.cache_resource(max_entries=1)
def load_model(
    model_dir,
    high_vram,
    kv_cache,
    ar_checkpoint,
    diff_checkpoint,
):
    gc.collect()
    return TextToSpeech(
        models_dir=model_dir,
        high_vram=high_vram,
        kv_cache=kv_cache,
        ar_checkpoint=ar_checkpoint,
        diff_checkpoint=diff_checkpoint,
    )


@st.cache_data
def list_voices(extra_voices_dir: Optional[str]):
    voices = ["random"]
    if extra_voices_dir and os.path.isdir(extra_voices_dir):
        voices.extend(os.listdir(extra_voices_dir))
        extra_voices_ls = [extra_voices_dir]
    else:
        extra_voices_ls = []
    voices.extend(
        [v for v in os.listdir("tortoise/voices") if v != "cond_latent_example"]
    )
    #
    return voices, extra_voices_ls


@st.cache_resource(max_entries=1)
def load_voice_conditionings(voice, extra_voices_ls):
    gc.collect()
    voice_samples, conditioning_latents = load_voices(voice, extra_voices_ls)
    return voice_samples, conditioning_latents
