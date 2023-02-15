import sys
import os
from random import randint
from typing import Optional
from tortoise.api import MODELS_DIR, TextToSpeech
from tortoise.utils.audio import get_voices, load_voices, load_audio
from tortoise.utils.text import split_and_recombine_text

def get_all_voices(extra_voice_dirs_str: str=''):
    extra_voice_dirs = extra_voice_dirs_str.split(',') if extra_voice_dirs_str else []
    return sorted(get_voices(extra_voice_dirs)), extra_voice_dirs

def parse_voice_str(voice_str: str, all_voices: list[str]):
    selected_voices = all_voices if voice_str == 'all' else voice_str.split(',')
    selected_voices = [v.split('&') if '&' in v else [v] for v in selected_voices]
    for voices in selected_voices:
        for v in voices:
            if v != 'random' and v not in all_voices:
                raise ValueError(f'voice {v} not available, use --list-voices to see available voices.')

    return selected_voices

def voice_loader(selected_voices: list, extra_voice_dirs: list[str]):
    for voices in selected_voices:
        yield voices, *load_voices(voices, extra_voice_dirs)

def parse_multiarg_text(text: list[str]):
    return (
        ' '.join(text) if text else
        ''.join(line for line in sys.stdin)
    ).strip()

def split_text(text: str, text_split: str):
    if text_split:
        desired_length, max_length = map(int, text_split.split(','))
        if desired_length > max_length:
            raise ValueError(f'--text-split: desired_length ({desired_length}) must be <= max_length ({max_length})')
        texts = split_and_recombine_text(text, desired_length, max_length)
    else:
        texts = split_and_recombine_text(text)
    #
    if not texts:
        raise ValueError('no text provided')
    return texts

def validate_output_dir(output_dir: str, selected_voices: list, candidates: int):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        if len(selected_voices) > 1:
            raise ValueError('cannot have multiple voices without --output-dir"')
        if candidates > 1:
            raise ValueError('cannot have multiple candidates without --output-dir"')
    return output_dir

def check_pydub(play: bool):
    if play:
        try:
            import pydub
            import pydub.playback
            return pydub
        except ImportError:
            raise RuntimeError('--play requires pydub to be installed, which can be done with "pip install pydub"')

def get_seed(seed: Optional[int]):
    return randint(0, 2**32-1) if seed is None else seed

    

