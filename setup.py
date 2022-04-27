from setuptools import setup, find_packages

install_requires = [
    "torch",
    "torchaudio",
    "rotary_embedding_torch",
    "transformers",
    "tokenizers",
    "inflect",
    "progressbar",
    "einops",
    "unidecode",
    "scipy",
    "librosa"
]

setup(
    name="tortoise_tts",
    packages=['tortoise_tts'],
    install_requires=install_requires,
)
