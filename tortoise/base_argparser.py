import argparse
from api import TextToSpeech, MODELS_DIR

from utils.diffusion import K_DIFFUSION_SAMPLERS
SAMPLERS = list(K_DIFFUSION_SAMPLERS.keys()) + ['ddim']

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
ap.add_argument('--kv_cache', help='enable (partially broken) kv_cache usage, leading to drastic speedups but worse memory usage + results', default=False, action='store_true')
ap.add_argument('--sampler', help='override the sampler used for diffusion (default depends on --preset)', choices=SAMPLERS)
ap.add_argument('--steps', type=int, help='override the steps used for diffusion (default depends on --preset)')
ap.add_argument('--cond_free', help='force conditioning free diffusion', action='store_true')
ap.add_argument('--no_cond_free', help='force disable conditioning free diffusion', dest='cond_free', action='store_false')
ap.add_argument('--cvvp_amount', type=float, help='How much the CVVP model should influence the output.'
                'Increasing this can in some cases reduce the likelihood of multiple speakers. Defaults to 0 (disabled)', default=.0)


