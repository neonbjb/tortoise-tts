curl "http://127.0.0.1:42110/task-result/$TASK_ID" \
     -H "Authorization: Bearer ${ACCESS_TOKEN}" \
     -H "Content-Type: application/json" \
     -o data/samples/curl-task-result.wav
echo "data/samples/curl-task-result.wav"