import argparse
import os

import torch
import torch.nn.functional as F
import torchaudio

from api import TextToSpeech, load_conditioning
from utils.audio import load_audio
from utils.tokenizer import VoiceBpeTokenizer

def split_and_recombine_text(texts, desired_length=200, max_len=300):
    # TODO: also split across '!' and '?'. Attempt to keep quotations together.
    texts = [s.strip() + "." for s in texts.split('.')]

    i = 0
    while i < len(texts):
        ltxt = texts[i]
        if len(ltxt) >= desired_length or i == len(texts)-1:
            i += 1
            continue
        if len(ltxt) + len(texts[i+1]) > max_len:
            i += 1
            continue
        texts[i] = f'{ltxt} {texts[i+1]}'
        texts.pop(i+1)
    return texts

if __name__ == '__main__':
    # These are voices drawn randomly from the training set. You are free to substitute your own voices in, but testing
    # has shown that the model does not generalize to new voices very well.
    preselected_cond_voices = {
        # Male voices
        'dotrice': ['voices/dotrice/1.wav', 'voices/dotrice/2.wav'],
        'harris': ['voices/harris/1.wav', 'voices/harris/2.wav'],
        'lescault': ['voices/lescault/1.wav', 'voices/lescault/2.wav'],
        'otto': ['voices/otto/1.wav', 'voices/otto/2.wav'],
        'obama': ['voices/obama/1.wav', 'voices/obama/2.wav'],
        'carlin': ['voices/carlin/1.wav', 'voices/carlin/2.wav'],
        # Female voices
        'atkins': ['voices/atkins/1.wav', 'voices/atkins/2.wav'],
        'grace': ['voices/grace/1.wav', 'voices/grace/2.wav'],
        'kennard': ['voices/kennard/1.wav', 'voices/kennard/2.wav'],
        'mol': ['voices/mol/1.wav', 'voices/mol/2.wav'],
        'lj': ['voices/lj/1.wav', 'voices/lj/2.wav'],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-textfile', type=str, help='A file containing the text to read.', default="data/riding_hood.txt")
    parser.add_argument('-voice', type=str, help='Use a preset conditioning voice (defined above). Overrides cond_path.', default='dotrice')
    parser.add_argument('-num_samples', type=int, help='How many total outputs the autoregressive transformer should produce.', default=512)
    parser.add_argument('-batch_size', type=int, help='How many samples to process at once in the autoregressive model.', default=16)
    parser.add_argument('-output_path', type=str, help='Where to store outputs.', default='results/longform/')
    parser.add_argument('-generation_preset', type=str, help='Preset to use for generation', default='intelligible')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    with open(args.textfile, 'r', encoding='utf-8') as f:
        text = ''.join([l for l in f.readlines()])
    texts = split_and_recombine_text(text)

    tts = TextToSpeech(autoregressive_batch_size=args.batch_size)

    priors = []
    for j, text in enumerate(texts):
        cond_paths = preselected_cond_voices[args.voice]
        conds = priors.copy()
        for cond_path in cond_paths:
            c = load_audio(cond_path, 22050)
            conds.append(c)
        gen = tts.tts_with_preset(text, conds, preset=args.generation_preset, num_autoregressive_samples=args.num_samples)
        torchaudio.save(os.path.join(args.output_path, f'{j}.wav'), gen.squeeze(0).cpu(), 24000)

        priors.append(torchaudio.functional.resample(gen, 24000, 22050).squeeze(0))
        while len(priors) > 2:
            priors.pop(0)

