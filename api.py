import argparse
import os
import random
from urllib import request

import torch
import torch.nn.functional as F
import progressbar

from models.diffusion_decoder import DiffusionTts
from models.autoregressive import UnifiedVoice
from tqdm import tqdm

from models.arch_util import TorchMelSpectrogram
from models.text_voice_clip import VoiceCLIP
from models.vocoder import UnivNetGenerator
from utils.audio import load_audio, wav_to_univnet_mel, denormalize_tacotron_mel
from utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from utils.tokenizer import VoiceBpeTokenizer, lev_distance


pbar = None
def download_models():
    MODELS = {
        'clip.pth': 'https://huggingface.co/jbetker/tortoise-tts-clip/resolve/main/pytorch-model.bin',
        'diffusion.pth': 'https://huggingface.co/jbetker/tortoise-tts-diffusion-v1/resolve/main/pytorch-model.bin',
        'autoregressive.pth': 'https://huggingface.co/jbetker/tortoise-tts-autoregressive/resolve/main/pytorch-model.bin'
    }
    os.makedirs('.models', exist_ok=True)
    def show_progress(block_num, block_size, total_size):
        global pbar
        if pbar is None:
            pbar = progressbar.ProgressBar(maxval=total_size)
            pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(downloaded)
        else:
            pbar.finish()
            pbar = None
    for model_name, url in MODELS.items():
        if os.path.exists(f'.models/{model_name}'):
            continue
        print(f'Downloading {model_name} from {url}...')
        request.urlretrieve(url, f'.models/{model_name}', show_progress)
        print('Done.')


def pad_or_truncate(t, length):
    if t.shape[-1] == length:
        return t
    elif t.shape[-1] < length:
        return F.pad(t, (0, length-t.shape[-1]))
    else:
        return t[..., :length]


def load_discrete_vocoder_diffuser(trained_diffusion_steps=4000, desired_diffusion_steps=200, cond_free=True, cond_free_k=1):
    """
    Helper function to load a GaussianDiffusion instance configured for use as a vocoder.
    """
    return SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=cond_free, conditioning_free_k=cond_free_k)


def load_conditioning(clip, cond_length=132300):
    gap = clip.shape[-1] - cond_length
    if gap < 0:
        clip = F.pad(clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = random.randint(0, gap)
        clip = clip[:, rand_start:rand_start + cond_length]
    mel_clip = TorchMelSpectrogram()(clip.unsqueeze(0)).squeeze(0)
    return mel_clip.unsqueeze(0).cuda()


def clip_guided_generation(autoregressive_model, clip_model, conditioning_input, text_input, num_batches, stop_mel_token,
                           tokens_per_clip_inference=10, clip_results_to_reduce_to=8, **generation_kwargs):
    """
    Uses a CLVP model trained to associate full text with **partial** audio clips to pick the best generation candidates
    every few iterations. The top results are then propagated forward through the generation process. Rinse and repeat.
    This is a hybrid between beam search and sampling.
    """
    token_goal = tokens_per_clip_inference
    finished = False
    while not finished and token_goal < autoregressive_model.max_mel_tokens:
        samples = []
        for b in tqdm(range(num_batches)):
            codes = autoregressive_model.inference_speech(conditioning_input, text_input, **generation_kwargs)
            samples.append(codes)
        for batch in samples:
            for i in range(batch.shape[0]):
                batch[i] = fix_autoregressive_output(batch[i], stop_mel_token, complain=False)
            clip_results.append(clip_model(text_input.repeat(batch.shape[0], 1), batch, return_loss=False))
        clip_results = torch.cat(clip_results, dim=0)
        samples = torch.cat(samples, dim=0)
        best_results = samples[torch.topk(clip_results, k=clip_results_to_reduce_to).indices]


def fix_autoregressive_output(codes, stop_token, complain=True):
    """
    This function performs some padding on coded audio that fixes a mismatch issue between what the diffusion model was
    trained on and what the autoregressive code generator creates (which has no padding or end).
    This is highly specific to the DVAE being used, so this particular coding will not necessarily work if used with
    a different DVAE. This can be inferred by feeding a audio clip padded with lots of zeros on the end through the DVAE
    and copying out the last few codes.

    Failing to do this padding will produce speech with a harsh end that sounds like "BLAH" or similar.
    """
    # Strip off the autoregressive stop token and add padding.
    stop_token_indices = (codes == stop_token).nonzero()
    if len(stop_token_indices) == 0:
        if complain:
            print("No stop tokens found, enjoy that output of yours!")
        return codes
    else:
        codes[stop_token_indices] = 83
    stm = stop_token_indices.min().item()
    codes[stm:] = 83
    if stm - 3 < codes.shape[0]:
        codes[-3] = 45
        codes[-2] = 45
        codes[-1] = 248

    return codes


def do_spectrogram_diffusion(diffusion_model, diffuser, mel_codes, conditioning_samples, temperature=1):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        cond_mels = []
        for sample in conditioning_samples:
            sample = pad_or_truncate(sample, 102400)
            cond_mel = wav_to_univnet_mel(sample.to(mel_codes.device), do_normalization=False)
            cond_mels.append(cond_mel)
        cond_mels = torch.stack(cond_mels, dim=1)

        output_seq_len = mel_codes.shape[1]*4*24000//22050  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        output_shape = (mel_codes.shape[0], 100, output_seq_len)
        precomputed_embeddings = diffusion_model.timestep_independent(mel_codes, cond_mels, output_seq_len, False)

        noise = torch.randn(output_shape, device=mel_codes.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                      model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings})
        return denormalize_tacotron_mel(mel)[:,:,:output_seq_len]


class TextToSpeech:
    def __init__(self, autoregressive_batch_size=16):
        self.autoregressive_batch_size = autoregressive_batch_size
        self.tokenizer = VoiceBpeTokenizer()
        download_models()

        self.autoregressive = UnifiedVoice(max_mel_tokens=604, max_text_tokens=402, max_conditioning_inputs=2, layers=30,
                                      model_dim=1024,
                                      heads=16, number_text_tokens=256, start_text_token=255, checkpointing=False,
                                      train_solo_embeddings=False,
                                      average_conditioning_embeddings=True).cpu().eval()
        self.autoregressive.load_state_dict(torch.load('.models/autoregressive.pth'))

        self.clip = VoiceCLIP(dim_text=512, dim_speech=512, dim_latent=512, num_text_tokens=256, text_enc_depth=12,
                             text_seq_len=350, text_heads=8,
                             num_speech_tokens=8192, speech_enc_depth=12, speech_heads=8, speech_seq_len=430,
                             use_xformers=True).cpu().eval()
        self.clip.load_state_dict(torch.load('.models/clip.pth'))

        self.diffusion = DiffusionTts(model_channels=1024, num_layers=10, in_channels=100, out_channels=200,
                                      in_latent_channels=1024, in_tokens=8193, dropout=0, use_fp16=False, num_heads=16,
                                      layer_drop=0, unconditioned_percentage=0).cpu().eval()
        self.diffusion.load_state_dict(torch.load('.models/diffusion_decoder.pth'))

        self.vocoder = UnivNetGenerator().cpu()
        self.vocoder.load_state_dict(torch.load('.models/vocoder.pth')['model_g'])
        self.vocoder.eval(inference=True)

    def tts_with_preset(self, text, voice_samples, preset='fast', **kwargs):
        """
        Calls TTS with one of a set of preset generation parameters. Options:
            'ultra_fast': Produces speech at a speed which belies the name of this repo. (Not really, but it's definitely fastest).
            'fast': Decent quality speech at a decent inference rate. A good choice for mass inference.
            'standard': Very good quality. This is generally about as good as you are going to get.
            'high_quality': Use if you want the absolute best. This is not really worth the compute, though.
        """
        # Use generally found best tuning knobs for generation.
        kwargs.update({'temperature': .8, 'length_penalty': 1.0, 'repetition_penalty': 2.0, 'top_p': .8,
                       'cond_free_k': 2.0, 'diffusion_temperature': 1.0})
        # Presets are defined here.
        presets = {
            'ultra_fast': {'num_autoregressive_samples': 32, 'diffusion_iterations': 16, 'cond_free': False},
            'fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 32},
            'standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 128},
            'high_quality': {'num_autoregressive_samples': 512, 'diffusion_iterations': 2048},
        }
        kwargs.update(presets[preset])
        return self.tts(text, voice_samples, **kwargs)

    def tts(self, text, voice_samples, k=1,
            # autoregressive generation parameters follow
            num_autoregressive_samples=512, temperature=.8, length_penalty=1, repetition_penalty=2.0, top_p=.8,
            # diffusion generation parameters follow
            diffusion_iterations=100, cond_free=True, cond_free_k=2, diffusion_temperature=1.0,):
        text = torch.IntTensor(self.tokenizer.encode(text)).unsqueeze(0).cuda()
        text = F.pad(text, (0, 1))  # This may not be necessary.

        conds = []
        if not isinstance(voice_samples, list):
            voice_samples = [voice_samples]
        for vs in voice_samples:
            conds.append(load_conditioning(vs))
        conds = torch.stack(conds, dim=1)

        diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=diffusion_iterations, cond_free=cond_free, cond_free_k=cond_free_k)

        with torch.no_grad():
            samples = []
            num_batches = num_autoregressive_samples // self.autoregressive_batch_size
            stop_mel_token = self.autoregressive.stop_mel_token
            calm_token = 83  # This is the token for coding silence, which is fixed in place with "fix_autoregressive_output"
            self.autoregressive = self.autoregressive.cuda()
            for b in tqdm(range(num_batches)):
                codes = self.autoregressive.inference_speech(conds, text,
                                                             do_sample=True,
                                                             top_p=top_p,
                                                             temperature=temperature,
                                                             num_return_sequences=self.autoregressive_batch_size,
                                                             length_penalty=length_penalty,
                                                             repetition_penalty=repetition_penalty)
                padding_needed = self.autoregressive.max_mel_tokens - codes.shape[1]
                codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
                samples.append(codes)
            self.autoregressive = self.autoregressive.cpu()

            clip_results = []
            self.clip = self.clip.cuda()
            for batch in samples:
                for i in range(batch.shape[0]):
                    batch[i] = fix_autoregressive_output(batch[i], stop_mel_token)
                clip_results.append(self.clip(text.repeat(batch.shape[0], 1), batch, return_loss=False))
            clip_results = torch.cat(clip_results, dim=0)
            samples = torch.cat(samples, dim=0)
            best_results = samples[torch.topk(clip_results, k=k).indices]
            self.clip = self.clip.cpu()
            del samples

            # The diffusion model actually wants the last hidden layer from the autoregressive model as conditioning
            # inputs. Re-produce those for the top results. This could be made more efficient by storing all of these
            # results, but will increase memory usage.
            self.autoregressive = self.autoregressive.cuda()
            best_latents = self.autoregressive(conds, text, torch.tensor([text.shape[-1]], device=conds.device), best_results,
                                               torch.tensor([best_results.shape[-1]*self.autoregressive.mel_length_compression], device=conds.device),
                                               return_latent=True, clip_inputs=False)
            self.autoregressive = self.autoregressive.cpu()

            print("Performing vocoding..")
            wav_candidates = []
            self.diffusion = self.diffusion.cuda()
            self.vocoder = self.vocoder.cuda()
            for b in range(best_results.shape[0]):
                codes = best_results[b].unsqueeze(0)
                latents = best_latents[b].unsqueeze(0)

                # Find the first occurrence of the "calm" token and trim the codes to that.
                ctokens = 0
                for k in range(codes.shape[-1]):
                    if codes[0, k] == calm_token:
                        ctokens += 1
                    else:
                        ctokens = 0
                    if ctokens > 8:  # 8 tokens gives the diffusion model some "breathing room" to terminate speech.
                        latents = latents[:, :k]
                        break

                mel = do_spectrogram_diffusion(self.diffusion, diffuser, latents, voice_samples, temperature=diffusion_temperature)
                wav = self.vocoder.inference(mel)
                wav_candidates.append(wav.cpu())
            self.diffusion = self.diffusion.cpu()
            self.vocoder = self.vocoder.cpu()

            if len(wav_candidates) > 1:
                return wav_candidates
            return wav_candidates[0]
