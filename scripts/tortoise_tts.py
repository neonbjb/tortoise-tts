#!/usr/bin/env python3
# AGPL: a notification must be added stating that changes have been made to that file. 

import argparse
import os
import sys
import tempfile
import time

import torch
import torchaudio

from tortoise.api import MODELS_DIR, TextToSpeech
from tortoise.utils.audio import get_voices, load_voices, load_audio
from tortoise.utils.text import split_and_recombine_text

parser = argparse.ArgumentParser(
    description='TorToiSe is a text-to-speech program that is capable of synthesizing speech '
                'in multiple voices with realistic prosody and intonation.')

parser.add_argument(
    'text', type=str, nargs='*',
    help='Text to speak. If omitted, text is read from stdin.')
parser.add_argument(
    '-v, --voice', type=str, default='random', metavar='VOICE', dest='voice',
    help='Selects the voice to use for generation. Use the & character to join two voices together. '
         'Use a comma to perform inference on multiple voices. Set to "all" to use all available voices. '
         'Note that multiple voices require the --output-dir option to be set.')
parser.add_argument(
    '-V, --voices-dir', metavar='VOICES_DIR', type=str, dest='voices_dir',
    help='Path to directory containing extra voices to be loaded. Use a comma to specify multiple directories.')
parser.add_argument(
    '-p, --preset', type=str, default='fast', choices=['ultra_fast', 'fast', 'standard', 'high_quality'], dest='preset',
    help='Which voice quality preset to use.')
parser.add_argument(
    '-q, --quiet', default=False, action='store_true', dest='quiet',
    help='Suppress all output.')

output_group = parser.add_mutually_exclusive_group(required=True)
output_group.add_argument(
    '-l, --list-voices', default=False, action='store_true', dest='list_voices',
    help='List available voices and exit.')
output_group.add_argument(
    '-P, --play', action='store_true', dest='play',
    help='Play the audio (requires pydub).')
output_group.add_argument(
    '-o, --output', type=str, metavar='OUTPUT', dest='output',
    help='Save the audio to a file.')
output_group.add_argument(
    '-O, --output-dir', type=str, metavar='OUTPUT_DIR', dest='output_dir',
    help='Save the audio to a directory as individual segments.')

multi_output_group = parser.add_argument_group('multi-output options (requires --output-dir)')
multi_output_group.add_argument(
    '--candidates', type=int, default=1,
    help='How many output candidates to produce per-voice. Note that only the first candidate is used in the combined output.')
multi_output_group.add_argument(
    '--regenerate', type=str, default=None,
    help='Comma-separated list of clip numbers to re-generate.')
multi_output_group.add_argument(
    '--skip-existing', action='store_true',
    help='Set to skip re-generating existing clips.')

advanced_group = parser.add_argument_group('advanced options')
advanced_group.add_argument(
    '--produce-debug-state', default=False, action='store_true',
    help='Whether or not to produce debug_states in current directory, which can aid in reproducing problems.')
advanced_group.add_argument(
    '--seed', type=int, default=None,
    help='Random seed which can be used to reproduce results.')
advanced_group.add_argument(
    '--models-dir', type=str, default=MODELS_DIR,
    help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to '
         '~/.cache/tortoise/.models, so this should only be specified if you have custom checkpoints.')
advanced_group.add_argument(
    '--text-split', type=str, default=None,
    help='How big chunks to split the text into, in the format <desired_length>,<max_length>.')
advanced_group.add_argument(
    '--disable-redaction', default=False, action='store_true',
    help='Normally text enclosed in brackets are automatically redacted from the spoken output '
         '(but are still rendered by the model), this can be used for prompt engineering. '
         'Set this to disable this behavior.')
advanced_group.add_argument(
    '--device', type=str, default=None,
    help='Device to use for inference.')
advanced_group.add_argument(
    '--batch-size', type=int, default=None,
    help='Batch size to use for inference. If omitted, the batch size is set based on available GPU memory.')

tuning_group = parser.add_argument_group('tuning options (overrides preset settings)')
tuning_group.add_argument(
    '--num-autoregressive-samples', type=int, default=None,
    help='Number of samples taken from the autoregressive model, all of which are filtered using CLVP. '
         'As TorToiSe is a probabilistic model, more samples means a higher probability of creating something "great".')
tuning_group.add_argument(
    '--temperature', type=float, default=None,
    help='The softmax temperature of the autoregressive model.')
tuning_group.add_argument(
    '--length-penalty', type=float, default=None,
    help='A length penalty applied to the autoregressive decoder. Higher settings causes the model to produce more terse outputs.')
tuning_group.add_argument(
    '--repetition-penalty', type=float, default=None,
    help='A penalty that prevents the autoregressive decoder from repeating itself during decoding. '
         'Can be used to reduce the incidence of long silences or "uhhhhhhs", etc.')
tuning_group.add_argument(
    '--top-p', type=float, default=None,
    help='P value used in nucleus sampling. 0 to 1. Lower values mean the decoder produces more "likely" (aka boring) outputs.')
tuning_group.add_argument(
    '--max-mel-tokens', type=int, default=None,
    help='Restricts the output length. 1 to 600. Each unit is 1/20 of a second.')
tuning_group.add_argument(
    '--cvvp-amount', type=float, default=None,
    help='How much the CVVP model should influence the output.'
    'Increasing this can in some cases reduce the likelihood of multiple speakers.')
tuning_group.add_argument(
    '--diffusion-iterations', type=int, default=None,
    help='Number of diffusion steps to perform.  More steps means the network has more chances to iteratively'
         'refine the output, which should theoretically mean a higher quality output. '
         'Generally a value above 250 is not noticeably better, however.')
tuning_group.add_argument(
    '--cond-free', type=bool, default=None,
    help='Whether or not to perform conditioning-free diffusion. Conditioning-free diffusion performs two forward passes for '
         'each diffusion step: one with the outputs of the autoregressive model and one with no conditioning priors. The output '
         'of the two is blended according to the cond_free_k value below. Conditioning-free diffusion is the real deal, and '
         'dramatically improves realism.')
tuning_group.add_argument(
    '--cond-free-k', type=float, default=None,
    help='Knob that determines how to balance the conditioning free signal with the conditioning-present signal. [0,inf]. '
         'As cond_free_k increases, the output becomes dominated by the conditioning-free signal. '
         'Formula is: output=cond_present_output*(cond_free_k+1)-cond_absenct_output*cond_free_k')
tuning_group.add_argument(
    '--diffusion-temperature', type=float, default=None,
    help='Controls the variance of the noise fed into the diffusion model. [0,1]. Values at 0 '
         'are the "mean" prediction of the diffusion network and will sound bland and smeared. ')

from tortoise.utils.diffusion import SAMPLERS
fast_group = parser.add_argument_group('new/speed options')
fast_group.add_argument('--low_vram', dest='high_vram', help='re-enable default offloading behaviour of tortoise', default=True, action='store_false')
fast_group.add_argument('--half', help='enable autocast to half precision for autoregressive model', default=False, action='store_true')
fast_group.add_argument('--kv_cache', help='no-op; kv_cache is enabled by default and this flag exists for compatibility', default=True, action='store_true')
fast_group.add_argument('--no_cache', help='disable kv_cache usage. This should really only be used if you are very low on vram.', action='store_false', dest='kv_cache')
fast_group.add_argument('--sampler', help='override the sampler used for diffusion (default depends on --preset)', choices=SAMPLERS)
fast_group.add_argument('--original_tortoise', help='ensure results are identical to original tortoise-tts repo', default=False, action='store_true')

usage_examples = f'''
Examples:

Read text using random voice and place it in a file:

    {parser.prog} -o hello.wav "Hello, how are you?"

Read text from stdin and play it using the tom voice:

    echo "Say it like you mean it!" | {parser.prog} -P -v tom

Read a text file using multiple voices and save the audio clips to a directory:

    {parser.prog} -O /tmp/tts-results -v tom,emma <textfile.txt
'''

from .inference import (
    get_all_voices, parse_voice_str, voice_loader,
    parse_multiarg_text, split_text,
    validate_output_dir, check_pydub, get_seed
)

# show usage even when Ctrl+C is pressed early
try:
    args = parser.parse_args()
except SystemExit as e:
    if e.code == 0:
        print(usage_examples)
    sys.exit(e.code)

# get voices 
all_voices, extra_voice_dirs = get_all_voices(args.voices_dir)
if args.list_voices:
    for v in all_voices: print(v)
    sys.exit(0)
selected_voices = parse_voice_str(args.voice, all_voices)
voice_generator = voice_loader(selected_voices, extra_voice_dirs)

# parse text
text = parse_multiarg_text(args.text)
texts = split_text(text, args.text_split)

output_dir = validate_output_dir(args.output_dir, selected_voices, args.candidates)

# error out early if pydub isn't installed
pydub = check_pydub(args.play)

seed = get_seed(args.seed)
verbose = not args.quiet

if verbose:
    print('Loading tts...')
tts = TextToSpeech(
    models_dir=args.models_dir, enable_redaction=not args.disable_redaction,
    device=args.device, autoregressive_batch_size=args.batch_size,
    high_vram=args.high_vram, kv_cache=args.kv_cache
)


gen_settings = {
    'use_deterministic_seed': seed,
    'verbose': verbose,
    'k': args.candidates,
    'preset': args.preset,
}
tuning_options = [
    'num_autoregressive_samples', 'temperature', 'length_penalty', 'repetition_penalty', 'top_p',
    'sampler', 'original_tortoise', 'half',
    'max_mel_tokens', 'cvvp_amount', 'diffusion_iterations', 'cond_free', 'cond_free_k', 'diffusion_temperature']
for option in tuning_options:
    if getattr(args, option) is not None:
        gen_settings[option] = getattr(args, option)
total_clips = len(texts) * len(selected_voices)
regenerate_clips = [int(x) for x in args.regenerate.split(',')] if args.regenerate else None
for voice_idx, (voice, voice_samples, conditioning_latents) in enumerate(voice_generator):
    audio_parts = []
    for text_idx, text in enumerate(texts):
        clip_name = f'{"-".join(voice)}_{text_idx:02d}'
        if args.output_dir:
            first_clip = os.path.join(args.output_dir, f'{clip_name}_00.wav')
            if (args.skip_existing or (regenerate_clips and text_idx not in regenerate_clips)) and os.path.exists(first_clip):
                audio_parts.append(load_audio(first_clip, 24000))
                if verbose:
                    print(f'Skipping {clip_name}')
                continue
        if verbose:
            print(f'Rendering {clip_name} ({(voice_idx * len(texts) + text_idx + 1)} of {total_clips})...')
            print('  ' + text)
        gen = tts.tts_with_preset(
            text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, **gen_settings)
        gen = gen if args.candidates > 1 else [gen]
        for candidate_idx, audio in enumerate(gen):
            audio = audio.squeeze(0).cpu()
            if candidate_idx == 0:
                audio_parts.append(audio)
            if args.output_dir:
                filename = f'{clip_name}_{candidate_idx:02d}.wav'
                torchaudio.save(os.path.join(args.output_dir, filename), audio, 24000)

    audio = torch.cat(audio_parts, dim=-1)
    if args.output_dir:
        filename = f'{"-".join(voice)}_combined.wav'
        torchaudio.save(os.path.join(args.output_dir, filename), audio, 24000)
    elif args.output:
        filename = args.output if args.output else os.tmp
        torchaudio.save(args.output, audio, 24000)
    elif args.play:
        f = tempfile.NamedTemporaryFile(suffix='.wav', delete=True)
        torchaudio.save(f.name, audio, 24000)
        pydub.playback.play(pydub.AudioSegment.from_wav(f.name))

    if args.produce_debug_state:
        os.makedirs('debug_states', exist_ok=True)
        dbg_state = (seed, texts, voice_samples, conditioning_latents, args)
        torch.save(dbg_state, os.path.join('debug_states', f'debug_{"-".join(voice)}.pth'))
