import argparse
import os
import sys


from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from beartype import beartype
from pydantic import BaseModel

from app.models.request import TranscriptionRequest
from app.models.tts import TTSArgs

from tortoise.do_tts import main as tts_main

app = FastAPI()

@beartype
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='Text to speak. This argument is required.')
    parser.add_argument('--voice', type=str, help="""
        Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
        'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.""", default='random')
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='fast')
    parser.add_argument('--output_path', type=str, help='Where to store outputs (directory).', default='data/results/')

    args = parser.parse_args()

    tts_args = TTSArgs(
        text=args.text,
        voice=args.voice,
        output_path=args.output_path,
        preset=args.preset,
    )

    try:
        output_path = local_inference_tts(tts_args)
        return output_path
    except Exception as e:
        print(f"Error during TTS generation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()