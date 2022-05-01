import argparse
import os

import torchaudio

from api import TextToSpeech
from tortoise.utils.audio import load_audio, get_voices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Text to speak.', default="I am a language model that has learned to speak.")
    parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
                                                 'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='pat')
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='standard')
    parser.add_argument('--voice_diversity_intelligibility_slider', type=float,
                        help='How to balance vocal diversity with the quality/intelligibility of the spoken text. 0 means highly diverse voice (not recommended), 1 means maximize intellibility',
                        default=.5)
    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/')
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
        gen = tts.tts_with_preset(args.text, conds, preset=args.preset, clvp_cvvp_slider=args.voice_diversity_intelligibility_slider)
        torchaudio.save(os.path.join(args.output_path, f'{voice}.wav'), gen.squeeze(0).cpu(), 24000)

