import pytest
import os, sys, time
import dotenv
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add the root directory of the repo to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tortoise')))

# load environment variables
load_envvar = dotenv.load_dotenv('tests/.env', override=True)
assert load_envvar and os.getenv("TESTING").lower()=="true", "Missing environment variables"

from app.main import app
client = TestClient(app)

# Helper function to simulate authentication
def basic_auth(username, password):
    import base64
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode("ascii")).decode("ascii")
    return {"Authorization": f"Basic {encoded_credentials}"}

@pytest.fixture(scope="module")
def task_id():
    with patch('app.main.text_to_speech') as mock_tts:
        mock_tts.return_value = {
            "task_id": "mock_task_id", "status": "queued"}
        request_data = {
            "text": "Hello, how are you?",
            "voice": "random",
            "preset": "ultra_fast"
        }
        with TestClient(app) as client:
            response = client.post(
                "/tts",
                headers=basic_auth(
                    os.getenv("TEST_USERNAME"), os.getenv("TEST_PASSWORD")),
                json=request_data
            )
            response.raise_for_status()
            return response.json().get("task_id")

@pytest.fixture(scope="module")
def access_token():
    with TestClient(app) as client:
        response = client.post(
            "/login",
            data={
                "username": os.getenv("TEST_USERNAME"),
                "password": os.getenv("TEST_PASSWORD")
            }
        )
        response.raise_for_status()
        token = response.json().get("access_token")

        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Response: {response.text}"
        assert "access_token" in response.json(), f"Access token missing in response. Response: {response.text}"
        assert "token_type" in response.json(), f"Token type missing in response. Response: {response.text}"
        assert response.json()["token_type"] == "bearer", f"Expected token type 'bearer', got {response.json()['token_type']}. Response: {response.text}"
        return token

def test_queue_status_api_key(access_token):
    with TestClient(app) as client:
        response = client.get(
            "/queue-status",
            headers={
                "Authorization": f"Bearer {access_token}"
            }
        )
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Response: {response.text}"
    assert 'tasks' in response.json(), f"'tasks' key missing in response. Response: {response.json()}"

def test_queue_status():
    with TestClient(app) as client:
        response = client.get(
            "/queue-status",
            headers=basic_auth(os.getenv("TEST_USERNAME"), os.getenv("TEST_PASSWORD"))
        )

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Response: {response.text}"
    assert 'tasks' in response.json(), f"'tasks' key missing in response. Response: {response.json()}"

def test_tts_task_creation(task_id):
    assert task_id is not None, f"Task ID is None. Check the task creation process."

    with TestClient(app) as client:
        response = client.get(
            "/queue-status",
            headers=basic_auth(
                os.getenv("TEST_USERNAME"), os.getenv("TEST_PASSWORD"))
        )
        response.raise_for_status()
        queue_status = response.json()
        assert task_id in queue_status["tasks"], f"Task ID {task_id} not found in queue tasks. Queue status: {queue_status}"
        assert queue_status["tasks"][task_id]["status"] in ["queued", "in_progress", "completed"],  f"Expected task status 'queued', got {queue_status['tasks'][task_id]['status']}. Queue status: {queue_status}"

def test_tts_task_completion(task_id):
    with patch('app.main.tasks') as mock_tasks:
        mock_tasks[task_id] = {
            "status": "completed",
            "result": {"message": "Task completed"}
        }

        with TestClient(app) as client:
            start_time = time.time()
            while True:
                response = client.get(
                    f"/task-status/{task_id}",
                    headers=basic_auth(
                        os.getenv("TEST_USERNAME"), os.getenv("TEST_PASSWORD"))
                )
                if response.status_code == 200 and response.json().get("status") == "completed":
                    break
                elif response.status_code == 500:
                    raise AssertionError(f"Task failed: {response.json().get('detail')}. Response: {response.text}")
                elif time.time() - start_time > 60:
                    raise AssertionError(f"Task did not complete within 1 minute. Task ID: {task_id}")
                time.sleep(5)

            # Verify that the task is completed
            assert response.status_code == 200, f"Expected status code 200, got {response.status_code}. Response: {response.text}"
            assert response.json()["status"] == "completed", f"Expected task status 'completed', got {response.json()['status']}. Response: {response.json()}"
            assert response.json()["result"]["message"] == "Task completed", f"Expected result message 'Task completed', got {response.json()['result']['message']}. Response: {response.json()}"

