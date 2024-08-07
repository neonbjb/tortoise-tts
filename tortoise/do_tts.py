import argparse
import logging
import logging.config
import os, sys
from tqdm import tqdm

import torch
import yaml
import torchaudio

# Add the root directory of the repo to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tortoise.api import TextToSpeech, MODELS_DIR, pick_best_batch_size_for_gpu
from tortoise.utils.audio import load_voices

# Configure logging to print to the console
# Load logging configuration
with open("logging_conf.yml", 'r') as file:
    config = yaml.safe_load(file.read())
    logging.config.dictConfig(config)
logger = logging.getLogger(__name__)

def main(args):
    # if torch.backends.mps.is_available():
    #     args.use_deepspeed = False
    os.makedirs(args.output_path, exist_ok=True)
    if not args.autoregressive_batch_size:
        args.autoregressive_batch_size = pick_best_batch_size_for_gpu()
    tts = TextToSpeech(
        models_dir=args.model_dir, autoregressive_batch_size=args.autoregressive_batch_size, use_deepspeed=args.use_deepspeed, kv_cache=args.kv_cache, half=args.half)

    selected_voices = args.voice.split(',')
    for k, selected_voice in tqdm(enumerate(selected_voices), desc="generating using selected voice"):
        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]
        voice_samples, conditioning_latents = load_voices(voice_sel)

        gen, dbg_state = tts.tts_with_preset(
            args.text, k=args.candidates, voice_samples=voice_samples, 
            conditioning_latents=conditioning_latents,
            preset=args.preset, use_deterministic_seed=args.seed, return_deterministic_state=True, cvvp_amount=args.cvvp_amount
        )
        if isinstance(gen, list):
            for j, g in enumerate(gen):
                output_path = os.path.join(args.output_path, f'{selected_voice}_{k}_{j}.wav')
                torchaudio.save(
                    output_path, g.squeeze(0).cpu(), 24000
                )
        else:
            output_path = os.path.join(args.output_path, f'{selected_voice}_{k}.wav')
            torchaudio.save(output_path, gen.squeeze(0).cpu(), 24000)
        print(f"Audio saved to {args.output_path} as {selected_voice}_{k}.wav")
        if args.produce_debug_state:
            os.makedirs('debug_states', exist_ok=True)
            torch.save(dbg_state, f'debug_states/do_tts_debug_{selected_voice}.pth')
        return output_path
            
if __name__ == '__main__':
    """
    class TTSArgs(BaseModel):
        text: str
        voice: str
        output_path: str
        preset: str
        model_dir: str = os.getenv("TORTOISE_MODELS_DIR", "data/models")
        use_deepspeed: bool = False
        kv_cache: bool = False
        half: bool = False
        candidates: int = 1
        seed: int = None
        cvvp_amount: float = 0.0
        produce_debug_state: bool = False
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'text', type=str, help='Text to speak. This argument is required.')
    parser.add_argument(
        '--voice', type=str, help="Selects the voice to use for generation. See options in voices/ directory (and add your own!) Use the & character to join two voices together. Use a comma to perform inference on multiple voices.", default='random')
    parser.add_argument(
        '--preset', type=str, help="""Which voice preset to use. Available presets = {
            'ultra_fast': {'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
            'fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
            'standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
            'high_quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400}
            }""", choices=['ultra_fast', 'fast', 'standard', 'high_quality'], default='fast')
    parser.add_argument(
        '--use_deepspeed', action=argparse.BooleanOptionalAction, type=bool, help='Use deepspeed for speed bump.', default=False)
    parser.add_argument(
        '--kv_cache', type=bool, action=argparse.BooleanOptionalAction, help='If you disable this please wait for a long a time to get the output', default=True)
    parser.add_argument(
        '--autoregressive_batch_size', type=int, help='Batch size for autoregressive inference.', default=pick_best_batch_size_for_gpu())
    parser.add_argument(
        '--half', type=bool, action=argparse.BooleanOptionalAction, help="float16(half) precision inference if True it's faster and take less vram and ram", default=True)
    parser.add_argument(
        '--output_path', type=str, help='Where to store outputs (directory).', default='data/samples/')
    parser.add_argument(
        '--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so thisshould only be specified if you have custom checkpoints.', default=MODELS_DIR)
    parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice.', default=3)
    parser.add_argument('--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
    parser.add_argument('--produce_debug_state', type=bool, action=argparse.BooleanOptionalAction, help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.', default=True)
    parser.add_argument('--cvvp_amount', type=float, help='How much the CVVP model should influence the output.'
                                                          'Increasing this can in some cases reduce the likelihood of multiple speakers. Defaults to 0 (disabled)', default=.0)
    args = parser.parse_args()
    assert args.half == False if args.use_deepspeed else True
    main(args)

