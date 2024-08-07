import os, sys
import tempfile
import logging
from fastapi.testclient import TestClient

# Add the root directory of the repo to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the tortoise directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tortoise')))

from app.models.tts import TTSArgs
from app.main import app

client = TestClient(app)

# Configure logging to print to the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_transcribe():
    logger.debug("Starting test_transcribe")
    # Create a temporary directory to store the output
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.debug(f"Created temporary directory at {temp_dir}")
        response = client.post(
            "/transcribe", json={
                "text": "Hello, this is a test.",
                "voice": "random",
                "output_path": temp_dir,
                "preset": "fast"
        })
        logger.debug(f"Received response with status code {response.status_code}")

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        files = os.listdir(temp_dir)
        assert len(files) >= 1  # or should equal args.candidate
        # check if ended in .wav
        assert files[0].endswith('.wav'), f"Expected the file to end with '.wav', but got {files[0]}"
        logger.debug("test_transcribe completed successfully")

def test_transcribe_file_not_found():
    logger.debug("Starting test_transcribe_file_not_found")
    # Create a temporary directory and delete it immediately to ensure the path does not exist
    temp_dir = tempfile.mkdtemp()
    logger.debug(f"Created temporary directory at {temp_dir}")
    os.rmdir(temp_dir)
    logger.debug(f"Deleted temporary directory at {temp_dir}")

    response = client.post(
        "/transcribe", json={
            "text": "Hello, this is a test.",
            "voice": "random",
            "output_path": temp_dir,
            "preset": "fast"
    })
    logger.debug(f"Received response with status code {response.status_code}")

    assert response.status_code == 404
    assert response.json() == {"detail": "File not found"}
    logger.debug("test_transcribe_file_not_found completed successfully")

def test_transcribe_internal_server_error(monkeypatch):
    logger.debug("Starting test_transcribe_internal_server_error")
    # Simulate an exception being raised in the `local_inference_tts` function
    def mock_local_inference_tts(args):
        logger.debug("Mock local_inference_tts called")
        raise Exception("Test exception")

    monkeypatch.setattr("app.main.local_inference_tts", mock_local_inference_tts)
    logger.debug("Replaced local_inference_tts with mock")

    response = client.post("/transcribe", json={
        "text": "Hello, this is a test.",
        "voice": "random",
        "output_path": "output.wav",
        "preset": "fast"
    })
    logger.debug(f"Received response with status code {response.status_code}")

    assert response.status_code == 500
    assert response.json() == {"detail": "Test exception"}
    logger.debug("test_transcribe_internal_server_error completed successfully")
