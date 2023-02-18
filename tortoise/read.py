# AGPL: a notification must be added stating that changes have been made to that file.

import argparse
import os
from time import time

import torch
from api import TextToSpeech
from base_argparser import ap, nullable_kwargs
from inference import save_gen_with_voicefix
from utils.audio import load_audio, load_voices
from utils.text import split_and_recombine_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[ap])
    parser.set_defaults(
        output_path="results/longform/",
        voice="pat",
        # preset='standard', #changed: preset is now fast
    )
    #
    parser.add_argument(
        "--textfile",
        type=str,
        help="A file containing the text to read.",
        default="tortoise/data/riding_hood.txt",
    )
    parser.add_argument(
        "--regenerate",
        type=str,
        help="Comma-separated list of clip numbers to re-generate, or nothing.",
        default=None,
    )
    parser.add_argument(
        "--candidates",
        type=int,
        help="How many output candidates to produce per-voice. Only the first candidate is actually used in the final product, the others can be used manually.",
        default=1,
    )
    # FORKED ARGUMENTS

    args = parser.parse_args()
    kwargs = nullable_kwargs(args)
    tts = TextToSpeech(
        models_dir=args.model_dir,
        high_vram=args.high_vram,
        kv_cache=args.kv_cache,
        ar_checkpoint=args.ar_checkpoint,
    )

    outpath = args.output_path
    selected_voices = args.voice.split(",")
    regenerate = args.regenerate
    if regenerate is not None:
        regenerate = [int(e) for e in regenerate.split(",")]

    # Process text
    with open(args.textfile, "r", encoding="utf-8") as f:
        text = " ".join([l for l in f.readlines()])
    if "|" in text:
        print(
            "Found the '|' character in your text, which I will use as a cue for where to split it up. If this was not"
            "your intent, please remove all '|' characters from the input."
        )
        texts = text.split("|")
    else:
        texts = split_and_recombine_text(text, 100, 200)

    seed = int(time()) if args.seed is None else args.seed
    for selected_voice in selected_voices:
        voice_outpath = os.path.join(outpath, selected_voice)
        os.makedirs(voice_outpath, exist_ok=True)

        if "&" in selected_voice:
            voice_sel = selected_voice.split("&")
        else:
            voice_sel = [selected_voice]

        voice_samples, conditioning_latents = load_voices(voice_sel)
        all_parts = []
        for j, text in enumerate(texts):
            if regenerate is not None and j not in regenerate:
                all_parts.append(
                    load_audio(os.path.join(voice_outpath, f"{j}.wav"), 24000)
                )
                continue
            gen = tts.tts_with_preset(
                text,
                voice_samples=voice_samples,
                conditioning_latents=conditioning_latents,
                preset=args.preset,
                k=args.candidates,
                use_deterministic_seed=seed,
                half=args.half,
                original_tortoise=args.original_tortoise,
                **kwargs,
            )
            if args.candidates == 1:
                save_gen_with_voicefix(gen, os.path.join(voice_outpath, f"{j}.wav"))
            else:
                candidate_dir = os.path.join(voice_outpath, str(j))
                os.makedirs(candidate_dir, exist_ok=True)
                for k, g in enumerate(gen):
                    save_gen_with_voicefix(g, os.path.join(candidate_dir, f"{k}.wav"))
                gen = gen[0].squeeze(0).cpu()
            all_parts.append(gen)

        if args.candidates == 1:
            full_audio = torch.cat(all_parts, dim=-1)
            save_gen_with_voicefix(
                full_audio, os.path.join(voice_outpath, "combined.wav"), squeeze=False
            )

        if args.produce_debug_state:
            os.makedirs("debug_states", exist_ok=True)
            dbg_state = (seed, texts, voice_samples, conditioning_latents)
            torch.save(dbg_state, f"debug_states/read_debug_{selected_voice}.pth")

        # Combine each candidate's audio clips.
        if args.candidates > 1:
            audio_clips = []
            for candidate in range(args.candidates):
                for line in range(len(texts)):
                    wav_file = os.path.join(
                        voice_outpath, str(line), f"{candidate}.wav"
                    )
                    audio_clips.append(load_audio(wav_file, 24000))
                audio_clips = torch.cat(audio_clips, dim=-1)
                save_gen_with_voicefix(
                    audio_clips,
                    os.path.join(voice_outpath, f"combined_{candidate:02d}.wav"),
                    squeeze=False,
                )
                audio_clips = []
