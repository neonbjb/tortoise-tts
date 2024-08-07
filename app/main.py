import os
import sys
import uuid
import asyncio
import dotenv

import concurrent.futures
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
from beartype import beartype

from app.models.request import TranscriptionRequest
from app.models.tts import TTSArgs
from app.services.oauth import get_current_user
from app.utils import pick_max_worker_function
from app.services.auth import get_current_username
from tortoise.do_tts import main as tts_main

dotenv.load_dotenv()
# Create the ThreadPoolExecutor with the determined number of max workers
max_workers = pick_max_worker_function()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

# Dictionary to store task information
tasks = {}

@beartype
def local_inference_tts(args: TTSArgs):
    """
    Run the TTS directly using the `main` function from `tortoise/do_tts.py`.
    Args:
    - args (TTSArgs): The arguments to pass to the TTS function.
    Returns:
    - str: Path to the output audio file.
    """
    output_path = tts_main(args)
    return output_path

async def process_requests():
    while True:
        task_id, (args, future) = await fifo_queue.get()  # Wait for a request from the queue
        try:
            tasks[task_id]['status'] = 'in_progress'
            output_path = await asyncio.get_event_loop().run_in_executor(executor, local_inference_tts, args)
            future.set_result(output_path)
            tasks[task_id]['status'] = 'completed'
            tasks[task_id]['result'] = output_path
        except Exception as e:
            future.set_exception(e)
            tasks[task_id]['status'] = 'failed'
            tasks[task_id]['error'] = str(e)
        finally:
            fifo_queue.task_done()  # Indicate that the request has been processed

@asynccontextmanager
async def lifespan(app: FastAPI):
    global fifo_queue
    fifo_queue = asyncio.Queue()  # Initialize the queue within the context
    # Start the request processing task
    task = asyncio.create_task(process_requests())
    yield
    # Clean up the task on shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

# Assign the lifespan context manager to the app
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def home():
    return JSONResponse(content={"message": "Hello, FiCast-TTS! Check the docs at /docs."})

@app.post("/transcribe", dependencies=[Depends(get_current_username)])
async def transcribe(request: TranscriptionRequest):
    try:
        args = TTSArgs(
            text=request.text,
            voice=request.voice,
            output_path=request.output_path,
            preset=request.preset
        )

        # Use a future to get the result of the inference
        future = asyncio.get_event_loop().create_future()
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        await fifo_queue.put((task_id, (args, future)))

        # Store task information
        tasks[task_id] = {
            'status': 'queued',
            'request': request,
            'result': None,
            'error': None
        }

        # Await the result of the future
        output_path = await future

        # Check if file exists
        if not os.path.isfile(output_path):
            raise HTTPException(status_code=404, detail=f"File not found: {output_path}")

        return FileResponse(output_path, media_type='audio/wav', filename=os.path.basename(output_path))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queue-status", dependencies=[Depends(get_current_username)])
async def queue_status():
    """
    Endpoint to get the current status of the queue.
    """
    return {
        "queue_length": fifo_queue.qsize(),
        "tasks": tasks
    }

@app.get("/task-status/{task_id}", dependencies=[Depends(get_current_username)])
async def task_status(task_id: str):
    """
    Endpoint to get the status of a specific task.
    """
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

if __name__ == "__main__":
    import argparse
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

        try:
            output_path = local_inference_tts(tts_args)
            print(f"Output stored at: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error during TTS generation: {str(e)}")
            sys.exit(1)
    main()
