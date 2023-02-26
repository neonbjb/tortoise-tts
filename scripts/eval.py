import os
from pathlib import Path
from typing import Literal

import torchaudio
from simple_parsing.decorators import main

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_required_audio


@main
def main(
    eval_path: Path,
    output_path: Path,
    preset: Literal["ultra_fast", "fast", "standard", "high_quality"] = "standard",
):
    """Evaluate preset quality

    Args:
        eval_path: Path to TSV test file
        output_path: Where to put the results
        preset: Rendering preset"""
    output_path.mkdir(exist_ok=True)

    tts = TextToSpeech()

    with eval_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        text, real = line.strip().split("\t")
        conds = [load_required_audio(real)]
        gen = tts.tts_with_preset(
            text, voice_samples=conds, conditioning_latents=None, preset=preset
        )
        torchaudio.save(
            output_path / os.path.basename(real),
            gen.squeeze(0).cpu(),
            24000,
        )


if __name__ == "__main__":
    main()
