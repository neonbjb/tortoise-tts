image="tts:app"
docker run --gpus all \
  -e TORTOISE_MODELS_DIR=/models \
  -v "${PWD}/data/models":/models \
  -v "${PWD}/data/results":/results \
  -v "${PWD}/data/.cache/huggingface":/root/.cache/huggingface \
  -p 42110:42110 \
  --name tts-api \
  -it $image --port 42110 --host 0.0.0.0