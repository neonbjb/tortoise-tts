import argparse
import os

import torch
import torchaudio

from api import TextToSpeech
from tortoise.utils.audio import load_audio, get_voices, load_voice

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Text to speak.', default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")
    parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
                                                 'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='random')
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='fast')
    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/')
    parser.add_argument('--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
                                                      'should only be specified if you have custom checkpoints.', default='.models')
    parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice.', default=3)
    parser.add_argument('--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
    parser.add_argument('--produce_debug_state', type=bool, help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.', default=True)
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    tts = TextToSpeech(models_dir=args.model_dir)

    selected_voices = args.voice.split(',')
    for k, voice in enumerate(selected_voices):
        voice_samples, conditioning_latents = load_voice(voice)
        gen, dbg_state = tts.tts_with_preset(args.text, k=args.candidates, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                  preset=args.preset, use_deterministic_seed=args.seed, return_deterministic_state=True)
        if isinstance(gen, list):
            for j, g in enumerate(gen):
                torchaudio.save(os.path.join(args.output_path, f'{voice}_{k}_{j}.wav'), g.squeeze(0).cpu(), 24000)
        else:
            torchaudio.save(os.path.join(args.output_path, f'{voice}_{k}.wav'), gen.squeeze(0).cpu(), 24000)

        if args.produce_debug_state:
            os.makedirs('debug_states', exist_ok=True)
            torch.save(dbg_state, f'debug_states/do_tts_debug_{voice}.pth')

