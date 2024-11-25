import torch
import torchaudio
import datetime
import tempfile
import gradio as gr
#import ffmpegio
# import numpy as np

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_voices

tts = TextToSpeech(models_dir=MODELS_DIR, use_deepspeed=False, kv_cache=True, half=True)

title = "TortoiseTTS UI"
description = "TUDDLE over Gradio"
article = "<p style='text-align: center'><a href='https://github.com/neonbjb/tortoise-tts' target='_blank' class='footer'>Github Repo</a></p>"

examples = [
]

# where is this coded in the tts generative model code?
sample_rate = 24000

def inference(speakers, text, seed, diterations):
    #get_debug_info = True if speakers == 'random' else False
    get_debug_info = True

    if ',' in speakers:
        voice_sel = speakers.split(',')
    else:
        voice_sel = [speakers]
    voice_samples, conditioning_latents = load_voices(voice_sel)

    if seed < 0:
        seed = None

    start = datetime.datetime.now()

    # k is how many samples to run
    # cvvp amount above 0 if you need to reduce multiple speakers
    # max_mel_tokens is the max number of 1/20 second length tokens used by something under the hood and impacts output duration. 500 ~= 25 seconds
    retval = tts.tts(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                             num_autoregressive_samples=96, diffusion_iterations=int(diterations), max_mel_tokens=500,
                             use_deterministic_seed=seed, cvvp_amount=0.0, return_deterministic_state=get_debug_info)
    debug_info = None
    conditioning_latents = None
    if get_debug_info:
        gen, debug_info = retval
        conditioning_latents = debug_info[4]
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as fp:
            torch.save(conditioning_latents, fp.name)
            debug_info = fp.name
    else:
        gen = retval

    if isinstance(gen, list):
        raise gr.Error("Keep k=1 to generate a single audio file.")

    audio_array = gen.squeeze(0).cpu()

    #with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fp:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        torchaudio.save(fp.name, audio_array, sample_rate)
        print(fp.name, debug_info)
        print("duration", datetime.datetime.now() - start)
        return (fp.name, debug_info)
        #with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3fp:
        #    ffmpegio.transcode(fp.name, mp3fp.name, overwrite=True)
        #    return mp3fp.name

gr.Interface(
    fn=inference,
    inputs=[
        gr.components.Textbox(
            label="Speaker",
            value="random"
        ),
        gr.components.Textbox(
            label="Text",
            value="Hello, my dog is cute",
        ),
        gr.components.Number(
            label="Seed",
            value=-1,
        ),
        gr.components.Number(
            label="DIterations",
            value=80,
            minimum=30,
            maximum=400,
        ),
    ],
    outputs=[
        gr.components.Audio(label="Speech", type="filepath"),
        gr.components.File(label="Latent from Random", type="file"),
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging='never',
    ).launch(debug=False, enable_queue=True, server_name="0.0.0.0")
