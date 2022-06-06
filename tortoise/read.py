import argparse
import os
from time import time

import torch
import torchaudio

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_audio, load_voices
from utils.text import split_and_recombine_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--textfile', type=str, help='A file containing the text to read.', default="tortoise/data/riding_hood.txt")
    parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
                                                 'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='pat')
    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/longform/')
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='standard')
    parser.add_argument('--regenerate', type=str, help='Comma-separated list of clip numbers to re-generate, or nothing.', default=None)
    parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice. Only the first candidate is actually used in the final product, the others can be used manually.', default=1)
    parser.add_argument('--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
                                                      'should only be specified if you have custom checkpoints.', default=MODELS_DIR)
    parser.add_argument('--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
    parser.add_argument('--produce_debug_state', type=bool, help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.', default=True)

    args = parser.parse_args()
    tts = TextToSpeech(models_dir=args.model_dir)

    outpath = args.output_path
    selected_voices = args.voice.split(',')
    regenerate = args.regenerate
    if regenerate is not None:
        regenerate = [int(e) for e in regenerate.split(',')]

    # Process text
    with open(args.textfile, 'r', encoding='utf-8') as f:
        text = ' '.join([l for l in f.readlines()])
    if '|' in text:
        print("Found the '|' character in your text, which I will use as a cue for where to split it up. If this was not"
              "your intent, please remove all '|' characters from the input.")
        texts = text.split('|')
    else:
        texts = split_and_recombine_text(text)

    seed = int(time()) if args.seed is None else args.seed
    for selected_voice in selected_voices:
        voice_outpath = os.path.join(outpath, selected_voice)
        os.makedirs(voice_outpath, exist_ok=True)

        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]

        voice_samples, conditioning_latents = load_voices(voice_sel)
        all_parts = []
        for j, text in enumerate(texts):
            if regenerate is not None and j not in regenerate:
                all_parts.append(load_audio(os.path.join(voice_outpath, f'{j}.wav'), 24000))
                continue
            gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                      preset=args.preset, k=args.candidates, use_deterministic_seed=seed)
            if args.candidates == 1:
                gen = gen.squeeze(0).cpu()
                torchaudio.save(os.path.join(voice_outpath, f'{j}.wav'), gen, 24000)
            else:
                candidate_dir = os.path.join(voice_outpath, str(j))
                os.makedirs(candidate_dir, exist_ok=True)
                for k, g in enumerate(gen):
                    torchaudio.save(os.path.join(candidate_dir, f'{k}.wav'), g.squeeze(0).cpu(), 24000)
                gen = gen[0].squeeze(0).cpu()
            all_parts.append(gen)

        if args.candidates == 1:
            full_audio = torch.cat(all_parts, dim=-1)
            torchaudio.save(os.path.join(voice_outpath, 'combined.wav'), full_audio, 24000)

        if args.produce_debug_state:
            os.makedirs('debug_states', exist_ok=True)
            dbg_state = (seed, texts, voice_samples, conditioning_latents)
            torch.save(dbg_state, f'debug_states/read_debug_{selected_voice}.pth')

        # Combine each candidate's audio clips.
        if args.candidates > 1:
            audio_clips = []
            for candidate in range(args.candidates):
                for line in range(len(texts)):
                    wav_file = os.path.join(voice_outpath, str(line), f"{candidate}.wav")
                    audio_clips.append(load_audio(wav_file, 24000))
                audio_clips = torch.cat(audio_clips, dim=-1)
                torchaudio.save(os.path.join(voice_outpath, f"combined_{candidate:02d}.wav"), audio_clips, 24000)
                audio_clips = []
