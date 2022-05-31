import setuptools
import os
import glob

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

data_dir = 'tortoise/data'
data_files = [(data_dir, list(glob.glob(os.path.join(data_dir, '*'))))]

setuptools.setup(
    name="TorToiSe",
    packages=setuptools.find_packages(),
    data_files=data_files,
    version="2.4.0",
    author="James Betker",
    author_email="james@adamant.ai",
    description="A high quality multi-voice text-to-speech library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neonbjb/tortoise-tts",
    project_urls={},
    scripts=[
        'scripts/tortoise_tts.py',
    ],
    install_requires=[
        'tqdm',
        'rotary_embedding_torch',
        'inflect',
        'progressbar',
        'einops',
        'unidecode',
        'scipy',
        'librosa',
        'transformers',
        'tokenizers',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)