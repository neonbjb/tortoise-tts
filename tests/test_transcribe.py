import pytest
import os
import sys
import logging
import asyncio
from fastapi.testclient import TestClient

# Add the root directory of the repo to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the tortoise directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tortoise')))

from app.main import app, tasks
from app.models.request import TranscriptionRequest

import dotenv
dotenv.load_dotenv()

# Configure logging to print to the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_fifo_queue():
    global fifo_queue
    fifo_queue = asyncio.Queue()

@pytest.fixture(scope="module")
def mocker():
    from pytest_mock import mocker
    return mocker

# Helper function to simulate authentication
def basic_auth(username, password):
    import base64
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode("ascii")).decode("ascii")
    return {"Authorization": f"Basic {encoded_credentials}"}

@pytest.mark.asyncio
async def test_transcribe(mocker):
    # Mock the local_inference_tts function to avoid running the actual TTS
    mocker.patch('app.main.local_inference_tts', return_value='data/tests/random_0.wav')
    request_data = {
        "text": "Hello, how are you?",
        "voice": "random",
        "output_path": "data/tests",
        "preset": "ultra_fast"
    }

    with TestClient(app) as client:
        response = client.post(
            "/tts",
            headers=basic_auth(os.getenv("TEST_USERNAME"), os.getenv("TEST_PASSWORD")),
            json=request_data
        )

    assert response.status_code == 200
    assert response.headers['content-type'] == 'audio/wav'
    # assert response.headers['content-disposition'] == 'attachment; filename=random_0.wav'

def test_queue_status():
    with TestClient(app) as client:
        response = client.get(
            "/queue-status",
            headers=basic_auth(os.getenv("TEST_USERNAME"), os.getenv("TEST_PASSWORD"))
        )

    assert response.status_code == 200
    assert 'queue_length' in response.json()
    assert 'tasks' in response.json()

def test_task_status():
    # Add a dummy task to the tasks dictionary
    task_id = 'test-task-id'
    tasks[task_id] = {
        'status': 'queued',
        'request': TranscriptionRequest(
            text="Hello, how are you?",
            voice="random",
            preset="fast"
        ),
        'result': None,
        'error': None
    }

    with TestClient(app) as client:
        response = client.get(
            f"/task-status/{task_id}",
            headers=basic_auth(os.getenv("TEST_USERNAME"), os.getenv("TEST_PASSWORD"))
        )

    assert response.status_code == 200
    assert response.json()['status'] == 'queued'

    # Clean up the tasks dictionary
    del tasks[task_id]

def test_failed_authentication():
    with TestClient(app) as client:
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