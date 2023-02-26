"""
Dumps the conditioning latents for the specified voice to disk. These are expressive latents which can be used for
other ML models, or can be augmented manually and fed back into Tortoise to affect vocal qualities.
"""
from pathlib import Path
from typing import Literal

import torch
from simple_parsing.decorators import main

from tortoise.api import TextToSpeech
from tortoise.utils.audio import get_voices, load_required_audio

EXAMPLE = Path(__file__ + "/../../results/conditioning_latents").resolve()


@main
def main(
    voice: str = "pat2",
    output_path: Path = EXAMPLE,
    latent_averaging_mode: Literal[0, 1, 2] = 0,
):
    """Dumps the conditioning latents for the specified voice to disk. These are expressive latents which can be used for
    other ML models, or can be augmented manually and fed back into Tortoise to affect vocal qualities.
    Args:
        voice: Selects the voice to convert to conditioning latents
        output_path: Where to store outputs.
        latent_averaging_mode: How to average voice latents, 0 for standard, 1 for per-sample, 2 for per-minichunk
    """
    output_path.mkdir(exist_ok=True)

    tts = TextToSpeech()
    voices = get_voices()
    selected_voices = voice.split(",")
    for voice in selected_voices:
        cond_paths = voices[voice]
        conds = []
        for cond_path in cond_paths:
            c = load_required_audio(cond_path)
            conds.append(c)
        conditioning_latents = tts.get_conditioning_latents(
            conds, latent_averaging_mode=latent_averaging_mode
        )
        torch.save(conditioning_latents, output_path / f"{voice}.pth")


if __name__ == "__main__":
    main()
