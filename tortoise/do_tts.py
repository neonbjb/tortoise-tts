# AGPL: a notification must be added stating that changes have been made to that file. 

import argparse
import os

import torch
import torchaudio

from api import TextToSpeech
from utils.audio import load_voices

from base_argparser import ap

from contextlib import contextmanager
from time import time
@contextmanager
def timeit(desc=''):
    start = time()
    yield
    print(f'{desc} took {time() - start:.2f} seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[ap])
    parser.add_argument('--text', type=str, help='Text to speak.', default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")
    parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice.', default=3)

    args = parser.parse_args()
    nullable_kwargs = {
        k:v for k,v in zip(
            ['sampler', 'diffusion_iterations', 'cond_free','num_autoregressive_samples'],
            [args.sampler, args.steps, args.cond_free, args.autoregressive_samples]
        ) if v is not None
    }
    os.makedirs(args.output_path, exist_ok=True)

    tts = TextToSpeech(models_dir=args.model_dir, high_vram=args.high_vram, kv_cache=args.kv_cache)

    selected_voices = args.voice.split(',')
    for k, selected_voice in enumerate(selected_voices):
        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]
        voice_samples, conditioning_latents = load_voices(voice_sel)

        with timeit(f'Generating {args.candidates} candidates for voice {selected_voice} (seed={args.seed})'):
            gen, dbg_state = tts.tts_with_preset(
                args.text, k=args.candidates, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                preset=args.preset, use_deterministic_seed=args.seed, return_deterministic_state=True, cvvp_amount=args.cvvp_amount,
                half=args.half, **nullable_kwargs
            )
        if isinstance(gen, list):
            for j, g in enumerate(gen):
                torchaudio.save(os.path.join(args.output_path, f'{selected_voice}_{k}_{j}.wav'), g.squeeze(0).cpu(), 24000)
        else:
            torchaudio.save(os.path.join(args.output_path, f'{selected_voice}_{k}.wav'), gen.squeeze(0).cpu(), 24000)

        if args.produce_debug_state:
            os.makedirs('debug_states', exist_ok=True)
            torch.save(dbg_state, f'debug_states/do_tts_debug_{selected_voice}.pth')

