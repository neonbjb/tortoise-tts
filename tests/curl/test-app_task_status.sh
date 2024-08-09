curl "http://127.0.0.1:42110/task-status/$TASK_ID" \
     -u $TEST_USERNAME:$TEST_PASSWORD \
     -H "Content-Type: application/json"
