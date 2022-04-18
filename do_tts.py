import argparse
import os

import torchaudio

from api import TextToSpeech
from utils.audio import load_audio, get_voices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Text to speak.', default="I am a language model that has learned to speak.")
    parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
                                                 'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='patrick_stewart')
    parser.add_argument('--num_samples', type=int, help='How many total outputs the autoregressive transformer should produce.', default=256)
    parser.add_argument('--batch_size', type=int, help='How many samples to process at once in the autoregressive model.', default=16)
    parser.add_argument('--num_diffusion_samples', type=int, help='Number of outputs that progress to the diffusion stage.', default=16)
    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    tts = TextToSpeech(autoregressive_batch_size=args.batch_size)

    voices = get_voices()
    selected_voices = args.voice.split(',')
    for voice in selected_voices:
        cond_paths = voices[voice]
        conds = []
        for cond_path in cond_paths:
            c = load_audio(cond_path, 22050)
            conds.append(c)
        gen = tts.tts(args.text, conds, num_autoregressive_samples=args.num_samples)
        torchaudio.save(os.path.join(args.output_path, f'{voice}.wav'), gen.squeeze(0).cpu(), 24000)

