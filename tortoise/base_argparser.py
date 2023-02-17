import argparse
from api import TextToSpeech, MODELS_DIR

from utils.diffusion import SAMPLERS

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
                                             'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='random')
ap.add_argument('--preset', type=str, help='Which voice preset to use.', default='fast')
ap.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/')
ap.add_argument('--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
                                                  'should only be specified if you have custom checkpoints.', default=MODELS_DIR)
ap.add_argument('--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
ap.add_argument('--produce_debug_state', type=bool, help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.', default=True)
# FORKED ARGUMENTS
ap.add_argument('--low_vram', dest='high_vram', help='re-enable default offloading behaviour of tortoise', default=True, action='store_false')
ap.add_argument('--half', help='enable autocast to half precision for autoregressive model', default=False, action='store_true')
ap.add_argument('--kv_cache', help='no-op; kv_cache is enabled by default and this flag exists for compatibility', default=True, action='store_true')
ap.add_argument('--no_cache', help='disable kv_cache usage. This should really only be used if you are very low on vram.', action='store_false', dest='kv_cache')
ap.add_argument('--sampler', help='override the sampler used for diffusion (default depends on --preset)', choices=SAMPLERS)
ap.add_argument('--steps', type=int, help='override the steps used for diffusion (default depends on --preset)')
ap.add_argument('--cond_free', type=bool, help='enable/disable conditioning free diffusion.')
ap.add_argument('--cvvp_amount', type=float, help='How much the CVVP model should influence the output.'
                'Increasing this can in some cases reduce the likelihood of multiple speakers. Defaults to 0 (disabled)', default=.0)
ap.add_argument('--autoregressive_samples', type=int, help='override the autoregressive_samples used for diffusion (default depends on --preset)')
ap.add_argument('--original_tortoise', help='ensure results are identical to original tortoise-tts repo', default=False, action='store_true')
ap.add_argument('--latent_averaging_mode', type=int, help='latent averaging mode (0/1/2)', default=0)
ap.add_argument('--ar-checkpoint', type=str, help='specific autoregressive model checkpoint to load over the default')

def nullable_kwargs(args, extras={}):
    mappings = {
        'sampler': 'sampler',
        'steps': 'diffusion_iterations',
        'cond_free': 'cond_free',
        'autoregressive_samples': 'num_autoregressive_samples',
        'latent_averaging_mode': 'latent_averaging_mode',
    }# | extras
    # for python3.8
    mappings = {**mappings, **extras}

    kwargs = {}
    for attr,arg in mappings.items():
        v = getattr(args, attr)
        if v is not None:
            kwargs[arg] = v

    return kwargs

