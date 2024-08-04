from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.models.request import TranscriptionRequest
from app.models.tts import TTSArgs

from tortoise.do_tts import main as tts_main

import os

app = FastAPI()

def local_inference_tts(args: TTSArgs):
    """
    Run the TTS directly using the `main` function from `tortoise/do_tts.py`.

    Args:
    - args (Args): The arguments to pass to the TTS function.

    Returns:
    - str: Path to the output audio file.
    """
    tts_main(args)
    return args.output_path
  
  
@app.post("/transcribe")
async def transcribe(request: TranscriptionRequest):
    try:
        args = TTSArgs(
            text=request.text,
            voice=request.voice,
            output_path=request.output_path,
            preset=request.preset
        )
        output_path = local_inference_tts(args)

        # Check if file exists
        if not os.path.isfile(output_path):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(output_path, media_type='audio/wav', filename=os.path.basename(output_path))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
