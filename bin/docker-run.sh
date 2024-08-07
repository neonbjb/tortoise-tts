image="tts:app"
docker run --gpus all \
  -e TORTOISE_MODELS_DIR=/models \
  -v "${PWD}/data/models":/models \
  -v "${PWD}/data/results":/results \
  -v "${PWD}/data/.cache/huggingface":/root/.cache/huggingface \
  -v /root:/work \
  --name tts-api \
  -it $image