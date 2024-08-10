response=$(curl -X POST "http://127.0.0.1:42110/tts" \
     -H "Authorization: Bearer ${ACCESS_TOKEN}" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Hello, curl test?",
           "voice": "random",
           "preset": "ultra_fast"
         }')
TASK_ID=$(echo $response | jq -r .task_id)
echo $TASK_ID

