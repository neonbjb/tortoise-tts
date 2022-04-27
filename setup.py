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
    packages=find_packages(),
    install_requires=install_requires,
)
