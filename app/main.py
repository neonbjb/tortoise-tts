import os
import sys
import uuid
import asyncio
import dotenv
import logging
import time
from datetime import timedelta

import concurrent.futures
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
from beartype import beartype
from fastapi.security import OAuth2PasswordRequestForm

from app.models.request import TranscriptionRequest
from app.models.tts import TTSArgs
# from app.services.oauth import get_current_user
from app.utils import pick_max_worker
from app.services.auth import get_current_user, verify_user, ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, timedelta

from tortoise.utils.audio import BUILTIN_VOICES_DIR
from tortoise.do_tts import _initialized_tts, infer_voice

load_envar = dotenv.load_dotenv()
assert load_envar and os.getenv("DEFAULT_USERNAME"), "Missing environment variables at .env"

# Environment-specific variable to skip initialization during testing
IS_TESTING = os.getenv("TESTING", "False").lower() in ("true", "1")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("app_debug.log"),  # Log to a file named `app_debug.log`
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Create the ThreadPoolExecutor with the determined number of max workers
executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("MAX_WORKERS", pick_max_worker())))
TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", 60*30))  # 30 minutes

# Dictionary to store task information
tasks = {}

@beartype
def local_inference_tts(tts, args):
    output_path = infer_voice(tts, args)
    return output_path

async def process_requests(tts):
    while True:
        task_id, (args, future) = await fifo_queue.get()
        try:
            tasks[task_id]['status'] = 'in_progress'
            output_path = await asyncio.get_event_loop().run_in_executor(executor, local_inference_tts, tts, args)
            future.set_result(output_path)
            tasks[task_id]['status'] = 'completed'
            tasks[task_id]['result'] = output_path
        except Exception as e:
            future.set_exception(e)
            tasks[task_id]['status'] = 'failed'
            tasks[task_id]['error'] = str(e)
        finally:
            fifo_queue.task_done()

async def post_initialization_event():
    try:
        request = TranscriptionRequest(
            text="Initialized! World", 
            voice="random", preset="ultra_fast"
        )
        response = await text_to_speech(request)
        print("Initialization TTS Response:", response)
    except Exception as e:
        print("Error during initialization TTS:", str(e))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global fifo_queue
    fifo_queue = asyncio.Queue()  # Initialize the queue within the context

    if not IS_TESTING:
        args = TTSArgs(text="")
        tts = _initialized_tts(args)
        task = asyncio.create_task(process_requests(tts))  # Start the request processing task
        await post_initialization_event()  # Post initialization event
    else:
        print(f"Skipping initialization due to TESTING={IS_TESTING}")
        task = None
    yield
    # Graceful shutdown
    if task:
        await fifo_queue.join()  # Wait for the queue to empty
        task.cancel()  # Cancel the task to exit the loop
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    # Shut down the executor
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def home():
    return JSONResponse(content={
        "message": "Hello, FiCast-TTS! Check the docs at /docs."})

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if verify_user(form_data.username, form_data.password):
        try:
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": form_data.username}, expires_delta=access_token_expires
            )
            print({"sub": form_data.username})
            print(access_token)
            return {
                "access_token": access_token, 
                "token_type": "bearer", 
                "user": form_data.username
            }
        except:
            raise HTTPException(
                status_code=401, detail="Unable to create access token")
    raise HTTPException(
        status_code=401, detail="Incorrect username or password")

@app.get("/voices")
async def available_voices():
    return JSONResponse(content={"voices": os.listdir(BUILTIN_VOICES_DIR)})

@app.post("/tts", dependencies=[Depends(get_current_user)])
async def text_to_speech(request: TranscriptionRequest):
    try:
        args = TTSArgs(
            text=request.text,
            voice=request.voice,
            preset=request.preset
        )

        future = asyncio.get_event_loop().create_future()
        task_id = str(uuid.uuid4())
        await fifo_queue.put((task_id, (args, future)))

        tasks[task_id] = {
            'status': 'queued',
            'request': request,
            'result': None,
            'error': None
        }
        return {"task_id": task_id, "status": "queued"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/queue-status", dependencies=[Depends(get_current_user)])
async def queue_status():
    try:
        return {
            "queue_length": fifo_queue.qsize(),
            "tasks": tasks
        }
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail, "headers": e.headers, "error": str(e)},
        )

@app.get("/task-status/{task_id}", dependencies=[Depends(get_current_user)])
async def task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/task-result/{task_id}", dependencies=[Depends(get_current_user)])
async def wait_for_result(task_id: str):
    try:
        task = tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        start_time = time.time()
        while task["status"] != "completed":
            if task["status"] == "failed":
                raise HTTPException(
                    status_code=500, detail=f"Task failed: {task.get('error', 'Unknown error')}")
            elif time.time() - start_time > TASK_TIMEOUT:
                raise HTTPException(status_code=408, detail="Task did not complete within the timeout")
            await asyncio.sleep(5)
            task = tasks.get(task_id)

        output_path = task["result"]
        if not os.path.isfile(output_path):
            raise HTTPException(status_code=404, detail=f"File not found: {output_path}")
        # Expected output
        return FileResponse(
            output_path, 
            filename=os.path.basename(output_path), 
            media_type="audio/wav")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import argparse
    import uvicorn
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('text', type=str, help='Text to speak. This argument is required.')
        parser.add_argument('--voice', type=str, help="""
            Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
            'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.""", default='random')
        parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='ultra_fast')

        args = parser.parse_args()

        tts_args = TTSArgs(
            text=args.text,
            voice=args.voice,
            preset=args.preset,
        )
        tts = _initialized_tts(tts_args)
        try:
            output_path = local_inference_tts(tts, tts_args)
            print(f"Output stored at: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error during TTS generation: {str(e)}")
            sys.exit(1)
            
    uvicorn.run(
        app, host="0.0.0.0", port=42110, log_level="debug")
