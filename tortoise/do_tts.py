# AGPL: a notification must be added stating that changes have been made to that file. 

import argparse
import os
from contextlib import contextmanager
from time import time

import torch
from api import TextToSpeech
from base_argparser import ap, nullable_kwargs
from inference import save_gen_with_voicefix
from utils.audio import load_voices


@contextmanager
def timeit(desc=''):
    start = time()
    yield
    print(f'{desc} took {time() - start:.2f} seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[ap])
    parser.add_argument('--text', type=str, help='Text to speak.', default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")
    parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice.', default=3)
    parser.add_argument('--voices-dir', type=str, help='extra voices dir')

    args = parser.parse_args()
    kwargs = nullable_kwargs(args)
    os.makedirs(args.output_path, exist_ok=True)

    tts = TextToSpeech(models_dir=args.model_dir, high_vram=args.high_vram, kv_cache=args.kv_cache, ar_checkpoint=args.ar_checkpoint)

    voices_dir = [args.voices_dir] if args.voices_dir else []
    selected_voices = args.voice.split(',')
    for k, selected_voice in enumerate(selected_voices):
        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]
        voice_samples, conditioning_latents = load_voices(voice_sel, voices_dir)

        with timeit(f'Generating {args.candidates} candidates for voice {selected_voice} (seed={args.seed})'):
            gen, dbg_state = tts.tts_with_preset(
                args.text, k=args.candidates, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                preset=args.preset, use_deterministic_seed=args.seed, return_deterministic_state=True, cvvp_amount=args.cvvp_amount,
                half=args.half, original_tortoise=args.original_tortoise, **kwargs
            )
        if isinstance(gen, list):
            for j, g in enumerate(gen):
                save_gen_with_voicefix(g, os.path.join(args.output_path, f'{selected_voice}_{k}_{j}.wav'))
        else:
            save_gen_with_voicefix(gen, os.path.join(args.output_path, f'{selected_voice}_{k}.wav'))

        if args.produce_debug_state:
            os.makedirs('debug_states', exist_ok=True)
            torch.save(dbg_state, f'debug_states/do_tts_debug_{selected_voice}.pth')

