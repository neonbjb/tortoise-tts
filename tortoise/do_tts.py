import argparse
import os

import torchaudio

from api import TextToSpeech
from tortoise.utils.audio import load_audio, get_voices, load_voice

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Text to speak.', default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")
    parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
                                                 'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='random')
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='fast')
    parser.add_argument('--voice_diversity_intelligibility_slider', type=float,
                        help='How to balance vocal diversity with the quality/intelligibility of the spoken text. 0 means highly diverse voice (not recommended), 1 means maximize intellibility',
                        default=.5)
    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/')
    parser.add_argument('--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
                                                      'should only be specified if you have custom checkpoints.', default='.models')
    parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice.', default=3)
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    tts = TextToSpeech(models_dir=args.model_dir)

    selected_voices = args.voice.split(',')
    for k, voice in enumerate(selected_voices):
        voice_samples, conditioning_latents = load_voice(voice)
        gen = tts.tts_with_preset(args.text, k=args.candidates, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                  preset=args.preset, clvp_cvvp_slider=args.voice_diversity_intelligibility_slider)
        if isinstance(gen, list):
            for j, g in enumerate(gen):
                torchaudio.save(os.path.join(args.output_path, f'{voice}_{k}_{j}.wav'), g.squeeze(0).cpu(), 24000)
        else:
            torchaudio.save(os.path.join(args.output_path, f'{voice}_{k}.wav'), gen.squeeze(0).cpu(), 24000)

