import argparse
import os
import torch

from api import TextToSpeech
from tortoise.utils.audio import load_audio, get_voices

"""
Dumps the conditioning latents for the specified voice to disk. These are expressive latents which can be used for
other ML models, or can be augmented manually and fed back into Tortoise to affect vocal qualities.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voice', type=str, help='Selects the voice to convert to conditioning latents', default='pat2')
    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='../results/conditioning_latents')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    tts = TextToSpeech()
    voices = get_voices()
    selected_voices = args.voice.split(',')
    for voice in selected_voices:
        cond_paths = voices[voice]
        conds = []
        for cond_path in cond_paths:
            c = load_audio(cond_path, 22050)
            conds.append(c)
        conditioning_latents = tts.get_conditioning_latents(conds)
        torch.save(conditioning_latents, os.path.join(args.output_path, f'{voice}.pth'))

