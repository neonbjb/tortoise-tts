import argparse
import os

import torchaudio

from api import TextToSpeech
from tortoise.utils.audio import load_audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', type=str, help='Path to TSV test file', default="D:\\tmp\\tortoise-tts-eval\\test.tsv")
    parser.add_argument('--output_path', type=str, help='Where to put results', default="D:\\tmp\\tortoise-tts-eval\\baseline")
    parser.add_argument('--preset', type=str, help='Rendering preset.', default="standard")
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    tts = TextToSpeech()

    with open(args.eval_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        text, real = line.strip().split('\t')
        conds = [load_audio(real, 22050)]
        gen = tts.tts_with_preset(text, voice_samples=conds, conditioning_latents=None, preset=args.preset)
        torchaudio.save(os.path.join(args.output_path, os.path.basename(real)), gen.squeeze(0).cpu(), 24000)

