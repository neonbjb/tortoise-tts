import argparse
import os

import torch
import torchaudio

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_voices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.", help='Text to speak.')
    parser.add_argument('--voice', type=str, default='random', help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) Use the & character to join two voices together. Use a comma to perform inference on multiple voices.')
    parser.add_argument('--preset', type=str, default='fast', help='Which voice preset to use.')
    parser.add_argument('--use_deepspeed', type=str, default=False, help='Which voice preset to use.')
    parser.add_argument('--output_path', type=str, default='results/', help='Where to store outputs.')
    parser.add_argument('--model_dir', type=str, default=MODELS_DIR, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this should only be specified if you have custom checkpoints.')
    parser.add_argument('--candidates', type=int, default=3, help='How many output candidates to produce per-voice.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed which can be used to reproduce results.')
    parser.add_argument('--produce_debug_state', type=bool, default=True, help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.')
    parser.add_argument('--cvvp_amount', type=float, default=.0, help='How much the CVVP model should influence the output. Increasing this can in some cases reduce the likelihood of multiple speakers. Defaults to 0 (disabled)')
    parser.add_argument('--models-dir', type=str, default=MODELS_DIR, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to ~/.cache/tortoise/.models, so this should only be specified if you have custom checkpoints.')
    parser.add_argument('--num_autoregressive_samples', type=int, default=None, help='Number of samples taken from the autoregressive model, all of which are filtered using CLVP. As TorToiSe is a probabilistic model, more samples means a higher probability of creating something "great".')
    parser.add_argument('--temperature', type=float, default=None, help='The softmax temperature of the autoregressive model.')
    parser.add_argument('--length_penalty', type=float, default=None, help='A length penalty applied to the autoregressive decoder. Higher settings causes the model to produce more terse outputs.')
    parser.add_argument('--repetition_penalty', type=float, default=None, help='A penalty that prevents the autoregressive decoder from repeating itself during decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc.')
    parser.add_argument('--top_p', type=float, default=None, help='P value used in nucleus sampling. 0 to 1. Lower values mean the decoder produces more "likely" (aka boring) outputs.')
    parser.add_argument('--max_mel_tokens', type=int, default=None, help='Restricts the output length. 1 to 600. Each unit is 1/20 of a second.')
    parser.add_argument('--diffusion_iterations', type=int, default=None, help='Number of diffusion steps to perform.  More steps means the network has more chances to iteratively refine the output, which should theoretically mean a higher quality output. Generally a value above 250 is not noticeably better, however.')
    parser.add_argument('--cond_free', type=bool, default=None, help='Whether or not to perform conditioning-free diffusion. Conditioning-free diffusion performs two forward passes for each diffusion step: one with the outputs of the autoregressive model and one with no conditioning priors. The output of the two is blended according to the cond_free_k value below. Conditioning-free diffusion is the real deal, and dramatically improves realism.')
    parser.add_argument('--cond_free_k', type=float, default=None, help='Knob that determines how to balance the conditioning free signal with the conditioning-present signal. [0,inf]. As cond_free_k increases, the output becomes dominated by the conditioning-free signal. Formula is: output=cond_present_output*(cond_free_k+1)-cond_absenct_output*cond_free_k')
    parser.add_argument('--diffusion_temperature', type=float, default=None, help='Controls the variance of the noise fed into the diffusion model. [0,1]. Values at 0 are the "mean" prediction of the diffusion network and will sound bland and smeared. ')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    tts = TextToSpeech(models_dir=args.model_dir, use_deepspeed=args.use_deepspeed)

    selected_voices = args.voice.split(',')
    for k, selected_voice in enumerate(selected_voices):
        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]
        voice_samples, conditioning_latents = load_voices(voice_sel)

#,seed="1234567", diffusion_temperature="0.2", top_p="1"
        gen, dbg_state = tts.tts_with_preset(
            args.text, k=args.candidates, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
            preset=args.preset, use_deterministic_seed=args.seed, return_deterministic_state=True, cvvp_amount=args.cvvp_amount,
            num_autoregressive_samples=args.num_autoregressive_samples, temperature=args.temperature,
            length_penalty=args.length_penalty, repetition_penalty=args.repetition_penalty,
            top_p=args.top_p, max_mel_tokens=args.max_mel_tokens, diffusion_iterations=args.diffusion_iterations,
            cond_free=args.cond_free, cond_free_k=args.cond_free_k, diffusion_temperature=args.diffusion_temperature
        )
        if isinstance(gen, list):
            for j, g in enumerate(gen):
                torchaudio.save(os.path.join(args.output_path, f'{selected_voice}_{k}_{j}.wav'), g.squeeze(0).cpu(), 24000)
        else:
            torchaudio.save(os.path.join(args.output_path, f'{selected_voice}_{k}.wav'), gen.squeeze(0).cpu(), 24000)

        if args.produce_debug_state:
            os.makedirs('debug_states', exist_ok=True)
            torch.save(dbg_state, f'debug_states/do_tts_debug_{selected_voice}.pth')

