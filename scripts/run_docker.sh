#!/bin/bash

set -e

docker build . -t tts

pwd=$(pwd)

docker run --gpus all \
    -e TORTOISE_MODELS_DIR=/models \
    -v $pwd/docker_data/models:/models \
    -v $pwd/docker_data/results:/results \
    -v $pwd/docker_data/.cache/huggingface:/root/.cache/huggingface \
    -v $pwd/docker_data/work:/work \
    -it tts
