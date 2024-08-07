import pytest
import os

from fastapi.testclient import TestClient
from app.main import app, tasks
from app.models.request import TranscriptionRequest

import dotenv
dotenv.load_dotenv()

client = TestClient(app)

# Helper function to simulate authentication
def basic_auth(username: str, password: str):
    import base64
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
    return {"Authorization": f"Basic {encoded_credentials}"}

# Test for /transcribe endpoint
@pytest.mark.asyncio
async def test_transcribe(mocker):
    # Mock the local_inference_tts function to avoid running the actual TTS
    mocker.patch('app.main.local_inference_tts', return_value='data/results/output.wav')
    request_data = {
        "text": "Hello, how are you?",
        "voice": "random",
        "output_path": "data/results",
        "preset": "fast"
    }

    response = client.post(
        "/transcribe",
        headers=basic_auth(os.getenv("TEST_USERNAME"), os.getenv("TEST_PASSWORD")),
        json=request_data
    )

    assert response.status_code == 200
    assert response.headers['content-type'] == 'audio/wav'
    assert response.headers['content-disposition'] == 'attachment; filename=output.wav'

# Test for /queue-status endpoint
def test_queue_status():
    response = client.get(
        "/queue-status",
        headers=basic_auth(os.getenv("TEST_USERNAME"), os.getenv("TEST_PASSWORD"))
    )

    assert response.status_code == 200
    assert 'queue_length' in response.json()
    assert 'tasks' in response.json()

# Test for /task-status/{task_id} endpoint
def test_task_status():
    # Add a dummy task to the tasks dictionary
    task_id = 'test-task-id'
    tasks[task_id] = {
        'status': 'queued',
        'request': TranscriptionRequest(
            text="Hello, how are you?",
            voice="random",
            output_path="data/tests",
            preset="fast"
        ),
        'result': None,
        'error': None
    }

    response = client.get(
        f"/task-status/{task_id}",
        headers=basic_auth(os.getenv("TEST_USERNAME"), os.getenv("TEST_PASSWORD"))
    )

    assert response.status_code == 200
    assert response.json()['status'] == 'queued'

    # Clean up the tasks dictionary
    del tasks[task_id]

# Test for failed authentication
def test_failed_authentication():
    response = client.post(
        "/transcribe",
        headers=basic_auth("wrong_username", "wrong_password"),
        json={
            "text": "Hello, how are you?",
            "voice": "random",
            "output_path": "data/tests",
            "preset": "fast"
        }
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect username or password"}
