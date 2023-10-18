import argparse
import os
from time import time

import torch
import torchaudio

from api_fast import TextToSpeech, MODELS_DIR
from utils.audio import load_audio, load_voices
from utils.text import split_and_recombine_text
import sounddevice as sd
import queue
import threading
def play_audio(audio_queue):
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        sd.play(chunk.cpu().numpy(), samplerate=24000)
        sd.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--textfile', type=str, help='A file containing the text to read.', default="tortoise/data/riding_hood.txt")
    parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
                                                 'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='lj')
    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/longform/')
    parser.add_argument('--output_name', type=str, help='How to name the output file', default='combined.wav')
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='standard')
    parser.add_argument('--regenerate', type=str, help='Comma-separated list of clip numbers to re-generate, or nothing.', default=None)
    parser.add_argument('--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
                                                      'should only be specified if you have custom checkpoints.', default=MODELS_DIR)
    parser.add_argument('--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
    parser.add_argument('--use_deepspeed', type=bool, help='Use deepspeed for speed bump.', default=False)
    parser.add_argument('--kv_cache', type=bool, help='If you disable this please wait for a long a time to get the output', default=True)
    parser.add_argument('--half', type=bool, help="float16(half) precision inference if True it's faster and take less vram and ram", default=True)


    args = parser.parse_args()
    if torch.backends.mps.is_available():
        args.use_deepspeed = False
    tts = TextToSpeech(models_dir=args.model_dir, use_deepspeed=args.use_deepspeed, kv_cache=args.kv_cache, half=args.half)

    outpath = args.output_path
    outname = args.output_name
    selected_voices = args.voice.split(',')
    regenerate = args.regenerate
    if regenerate is not None:
        regenerate = [int(e) for e in regenerate.split(',')]

    # Process text
    with open(args.textfile, 'r', encoding='utf-8') as f:
        text = ' '.join([l for l in f.readlines()])
    if '|' in text:
        print("Found the '|' character in your text, which I will use as a cue for where to split it up. If this was not"
              "your intent, please remove all '|' characters from the input.")
        texts = text.split('|')
    else:
        texts = split_and_recombine_text(text)
    audio_queue = queue.Queue()
    playback_thread = threading.Thread(target=play_audio, args=(audio_queue,))
    playback_thread.start()
    
    seed = int(time()) if args.seed is None else args.seed
    for selected_voice in selected_voices:
        voice_outpath = os.path.join(outpath, selected_voice)
        os.makedirs(voice_outpath, exist_ok=True)

        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]

        voice_samples, conditioning_latents = load_voices(voice_sel)
        all_parts = []
        for j, text in enumerate(texts):
            if regenerate is not None and j not in regenerate:
                all_parts.append(load_audio(os.path.join(voice_outpath, f'{j}.wav'), 24000))
                continue
            start_time = time()
            audio_generator = tts.tts_stream(text, voice_samples=voice_samples, use_deterministic_seed=seed)
            for wav_chunk in audio_generator:
                audio_queue.put(wav_chunk)
    audio_queue.put(None)
    playback_thread.join()