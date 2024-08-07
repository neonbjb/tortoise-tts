curl -X POST "http://127.0.0.1:42110/transcribe" \
     -u ficast-uzer:ficast-testpazz \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Hello, how are you?",
           "voice": "random",
           "preset": "ultra_fast"
         }' \
      -o data/samples/api-output.wav
         
