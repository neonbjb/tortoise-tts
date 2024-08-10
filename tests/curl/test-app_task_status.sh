curl "http://127.0.0.1:42110/task-status/$TASK_ID" \
     -H "Authorization: Bearer ${ACCESS_TOKEN}" \
     -H "Content-Type: application/json"
