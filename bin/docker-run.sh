docker run --gpus all \
  -e TORTOISE_MODELS_DIR=/models \
  -v ./data/models:/models \
  -v ./data/results:/results \
  -v ./data/.cache/huggingface:/root/.cache/huggingface \
  -v /root:/work \
  --name tts-app \
  -it tts