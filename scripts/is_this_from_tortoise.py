from pathlib import Path

from simple_parsing.decorators import main

from tortoise.api import classify_audio_clip
from tortoise.utils.audio import load_audio

EXAMPLE = Path(__file__ + "/../../examples/favorite_riding_hood.mp3").resolve()


@main
def main(clip: Path = EXAMPLE):
    """Find out whether a clip was generated with tortoise

    Args:
        clip: Path to an audio clip to classify."""

    clip = load_audio(str(clip), 24000)
    clip = clip[:, :220000]
    prob = classify_audio_clip(clip)
    print(
        f"This classifier thinks there is a {prob*100}% chance that this clip was generated from Tortoise."
    )


if __name__ == "__main__":
    main()
