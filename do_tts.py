import argparse
import os

import torch
import torch.nn.functional as F
import torchaudio

from api_new_autoregressive import TextToSpeech, load_conditioning
from utils.audio import load_audio
from utils.tokenizer import VoiceBpeTokenizer

if __name__ == '__main__':
    # These are voices drawn randomly from the training set. You are free to substitute your own voices in, but testing
    # has shown that the model does not generalize to new voices very well.
    preselected_cond_voices = {
        # Male voices
        'dotrice': ['voices/dotrice/1.wav', 'voices/dotrice/2.wav'],
        'harris': ['voices/harris/1.wav', 'voices/harris/2.wav'],
        'lescault': ['voices/lescault/1.wav', 'voices/lescault/2.wav'],
        'otto': ['voices/otto/1.wav', 'voices/otto/2.wav'],
        # Female voices
        'atkins': ['voices/atkins/1.wav', 'voices/atkins/2.wav'],
        'grace': ['voices/grace/1.wav', 'voices/grace/2.wav'],
        'kennard': ['voices/kennard/1.wav', 'voices/kennard/2.wav'],
        'mol': ['voices/mol/1.wav', 'voices/mol/2.wav'],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-text', type=str, help='Text to speak.', default="I am a language model that has learned to speak.")
    parser.add_argument('-voice', type=str, help='Use a preset conditioning voice (defined above). Overrides cond_path.', default='dotrice,harris,lescault,otto,atkins,grace,kennard,mol')
    parser.add_argument('-num_samples', type=int, help='How many total outputs the autoregressive transformer should produce.', default=32)
    parser.add_argument('-batch_size', type=int, help='How many samples to process at once in the autoregressive model.', default=16)
    parser.add_argument('-num_diffusion_samples', type=int, help='Number of outputs that progress to the diffusion stage.', default=16)
    parser.add_argument('-output_path', type=str, help='Where to store outputs.', default='results/')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    tts = TextToSpeech(autoregressive_batch_size=args.batch_size)

    for voice in args.voice.split(','):
        tokenizer = VoiceBpeTokenizer()
        text = torch.IntTensor(tokenizer.encode(args.text)).unsqueeze(0).cuda()
        text = F.pad(text, (0,1))  # This may not be necessary.
        cond_paths = preselected_cond_voices[voice]
        conds = []
        for cond_path in cond_paths:
            c = load_audio(cond_path, 22050)
            conds.append(c)
        gen = tts.tts(args.text, conds, num_autoregressive_samples=args.num_samples)
        torchaudio.save(os.path.join(args.output_path, f'{voice}.wav'), gen.squeeze(0).cpu(), 24000)

