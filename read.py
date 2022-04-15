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
        'emma_stone': ['voices/emma_stone/1.wav','voices/emma_stone/2.wav','voices/emma_stone/3.wav'],
        'tom_hanks': ['voices/tom_hanks/1.wav','voices/tom_hanks/2.wav','voices/tom_hanks/3.wav'],
        'patrick_stewart': ['voices/patrick_stewart/1.wav','voices/patrick_stewart/2.wav','voices/patrick_stewart/3.wav','voices/patrick_stewart/4.wav'],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-textfile', type=str, help='A file containing the text to read.', default="data/riding_hood.txt")
    parser.add_argument('-voice', type=str, help='Use a preset conditioning voice (defined above). Overrides cond_path.', default='patrick_stewart')
    parser.add_argument('-num_samples', type=int, help='How many total outputs the autoregressive transformer should produce.', default=128)
    parser.add_argument('-batch_size', type=int, help='How many samples to process at once in the autoregressive model.', default=16)
    parser.add_argument('-output_path', type=str, help='Where to store outputs.', default='results/longform/')
    parser.add_argument('-generation_preset', type=str, help='Preset to use for generation', default='realistic')
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

