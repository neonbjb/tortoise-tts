import argparse
import os
import random
from urllib import request

import torch
import torch.nn.functional as F
import torchaudio
import progressbar
import ocotillo

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


def fix_autoregressive_output(codes, stop_token):
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


def do_spectrogram_diffusion(diffusion_model, diffuser, mel_codes, conditioning_input, temperature=1):
    """
    Uses the specified diffusion model and DVAE model to convert the provided MEL & conditioning inputs into an audio clip.
    """
    with torch.no_grad():
        cond_mel = wav_to_univnet_mel(conditioning_input.squeeze(1), do_normalization=False)
        # Pad MEL to multiples of 32
        msl = mel_codes.shape[-1]
        dsl = 32
        gap = dsl - (msl % dsl)
        if gap > 0:
            mel = torch.nn.functional.pad(mel_codes, (0, gap))

        output_shape = (mel.shape[0], 100, mel.shape[-1]*4)
        precomputed_embeddings = diffusion_model.timestep_independent(mel_codes, cond_mel)

        noise = torch.randn(output_shape, device=mel_codes.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                      model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings})
        return denormalize_tacotron_mel(mel)[:,:,:msl*4]


class TextToSpeech:
    def __init__(self, autoregressive_batch_size=32):
        self.autoregressive_batch_size = autoregressive_batch_size
        self.tokenizer = VoiceBpeTokenizer()
        download_models()

        self.autoregressive = UnifiedVoice(max_mel_tokens=300, max_text_tokens=200, max_conditioning_inputs=2, layers=30,
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

        self.diffusion = DiffusionTts(model_channels=512, in_channels=100, out_channels=200, in_latent_channels=1024,
                                 channel_mult=[1, 2, 3, 4], num_res_blocks=[3, 3, 3, 3],
                                 token_conditioning_resolutions=[1, 4, 8],
                                 dropout=0, attention_resolutions=[4, 8], num_heads=8, kernel_size=3, scale_factor=2,
                                 time_embed_dim_multiplier=4, unconditioned_percentage=0, conditioning_dim_factor=2,
                                 conditioning_expansion=1).cpu().eval()
        self.diffusion.load_state_dict(torch.load('.models/diffusion.pth'))

        self.vocoder = UnivNetGenerator().cpu()
        self.vocoder.load_state_dict(torch.load('.models/vocoder.pth')['model_g'])
        self.vocoder.eval(inference=True)

    def tts(self, text, voice_samples, k=1,
            # autoregressive generation parameters follow
            num_autoregressive_samples=512, temperature=.9, length_penalty=1, repetition_penalty=1.0, top_k=50, top_p=.95,
            typical_sampling=False, typical_mass=.9,
            # diffusion generation parameters follow
            diffusion_iterations=100, cond_free=True, cond_free_k=1, diffusion_temperature=1,):
        text = torch.IntTensor(self.tokenizer.encode(text)).unsqueeze(0).cuda()
        text = F.pad(text, (0, 1))  # This may not be necessary.

        conds = []
        if not isinstance(voice_samples, list):
            voice_samples = [voice_samples]
        for vs in voice_samples:
            conds.append(load_conditioning(vs))
        conds = torch.stack(conds, dim=1)
        cond_diffusion = voice_samples[0].cuda()
        # The diffusion model expects = 88200 conditioning samples.
        if cond_diffusion.shape[-1] < 88200:
            cond_diffusion = F.pad(cond_diffusion, (0, 88200-cond_diffusion.shape[-1]))
        else:
            cond_diffusion = cond_diffusion[:, :88200]

        diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=diffusion_iterations, cond_free=cond_free, cond_free_k=cond_free_k)

        with torch.no_grad():
            samples = []
            num_batches = num_autoregressive_samples // self.autoregressive_batch_size
            stop_mel_token = self.autoregressive.stop_mel_token
            self.autoregressive = self.autoregressive.cuda()
            for b in tqdm(range(num_batches)):
                codes = self.autoregressive.inference_speech(conds, text,
                                                             do_sample=True,
                                                             top_k=top_k,
                                                             top_p=top_p,
                                                             temperature=temperature,
                                                             num_return_sequences=self.autoregressive_batch_size,
                                                             length_penalty=length_penalty,
                                                             repetition_penalty=repetition_penalty,
                                                             typical_sampling=typical_sampling,
                                                             typical_mass=typical_mass)
                padding_needed = 250 - codes.shape[1]
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

            print("Performing vocoding..")
            wav_candidates = []
            self.diffusion = self.diffusion.cuda()
            self.vocoder = self.vocoder.cuda()
            for b in range(best_results.shape[0]):
                code = best_results[b].unsqueeze(0)
                mel = do_spectrogram_diffusion(self.diffusion, diffuser, code, cond_diffusion, temperature=diffusion_temperature)
                wav = self.vocoder.inference(mel)
                wav_candidates.append(wav.cpu())
            self.diffusion = self.diffusion.cpu()
            self.vocoder = self.vocoder.cpu()

            if len(wav_candidates) > 1:
                return wav_candidates
            return wav_candidates[0]