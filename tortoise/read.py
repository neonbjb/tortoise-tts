# AGPL: a notification must be added stating that changes have been made to that file. 

import argparse
import os
from time import time

import torch
import torchaudio

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_audio, load_voices
from utils.text import split_and_recombine_text

from base_argparser import ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[ap])
    parser.set_defaults(
        output_path='results/longform/',
        voice='pat',
        #preset='standard', #changed: preset is now fast
    )
    #
    parser.add_argument('--textfile', type=str, help='A file containing the text to read.', default="tortoise/data/riding_hood.txt")
    parser.add_argument('--regenerate', type=str, help='Comma-separated list of clip numbers to re-generate, or nothing.', default=None)
    parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice. Only the first candidate is actually used in the final product, the others can be used manually.', default=1)
    # FORKED ARGUMENTS

    args = parser.parse_args()
    nullable_kwargs = {
        k:v for k,v in zip(
            ['sampler', 'diffusion_iterations', 'cond_free','num_autoregressive_samples'],
            [args.sampler, args.steps, args.cond_free, args.autoregressive_samples]
        ) if v is not None
    }
    tts = TextToSpeech(models_dir=args.model_dir, high_vram=args.high_vram, kv_cache=args.kv_cache)

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
        texts = split_and_recombine_text(text,100,200)

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
            gen = tts.tts_with_preset(
                text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                preset=args.preset, k=args.candidates, use_deterministic_seed=seed,
                half=args.half, original_tortoise=args.original_tortoise, **nullable_kwargs
            )
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
