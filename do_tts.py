import argparse
import os
import random
from urllib import request

import torch
import torch.nn.functional as F
import torchaudio
import progressbar

from models.diffusion_decoder import DiffusionTts
from models.autoregressive import UnifiedVoice
from tqdm import tqdm

from models.arch_util import TorchMelSpectrogram
from models.text_voice_clip import VoiceCLIP
from models.vocoder import UnivNetGenerator
from utils.audio import load_audio, wav_to_univnet_mel, denormalize_tacotron_mel
from utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from utils.tokenizer import VoiceBpeTokenizer

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


def load_discrete_vocoder_diffuser(trained_diffusion_steps=4000, desired_diffusion_steps=200):
    """
    Helper function to load a GaussianDiffusion instance configured for use as a vocoder.
    """
    return SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=True, conditioning_free_k=1)


def load_conditioning(path, sample_rate=22050, cond_length=132300):
    rel_clip = load_audio(path, sample_rate)
    gap = rel_clip.shape[-1] - cond_length
    if gap < 0:
        rel_clip = F.pad(rel_clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = random.randint(0, gap)
        rel_clip = rel_clip[:, rand_start:rand_start + cond_length]
    mel_clip = TorchMelSpectrogram()(rel_clip.unsqueeze(0)).squeeze(0)
    return mel_clip.unsqueeze(0).cuda(), rel_clip.unsqueeze(0).cuda()


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
        return
    else:
        codes[stop_token_indices] = 83
    stm = stop_token_indices.min().item()
    codes[stm:] = 83
    if stm - 3 < codes.shape[0]:
        codes[-3] = 45
        codes[-2] = 45
        codes[-1] = 248

    return codes


def do_spectrogram_diffusion(diffusion_model, diffuser, mel_codes, conditioning_input, mean=False):
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
        if mean:
            mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=torch.zeros(output_shape, device=mel_codes.device),
                                          model_kwargs={'aligned_conditioning': mel_codes, 'conditioning_input': cond_mel})
        else:
            mel = diffuser.p_sample_loop(diffusion_model, output_shape, model_kwargs={'aligned_conditioning': mel_codes, 'conditioning_input': cond_mel})
        return denormalize_tacotron_mel(mel)[:,:,:msl*4]


if __name__ == '__main__':
    # These are voices drawn randomly from the training set. You are free to substitute your own voices in, but testing
    # has shown that the model does not generalize to new voices very well.
    preselected_cond_voices = {
        # Male voices
        'dotrice': ['voices/dotrice/1.wav', 'voices/dotrice/2.wav'],
        'harris': ['voices/harris/1.wav', 'voices/harris/2.wav'],
        'lescault': ['voices/lescault/1.wav', 'voices/lescault/2.wav'],
        'otto': ['voices/otto/1.wav', 'voices/otto/2.wav'],
        # Female voices
        'atkins': ['voices/atkins/1.wav', 'voices/atkins/2.wav'],
        'grace': ['voices/grace/1.wav', 'voices/grace/2.wav'],
        'kennard': ['voices/kennard/1.wav', 'voices/kennard/2.wav'],
        'mol': ['voices/mol/1.wav', 'voices/mol/2.wav'],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-text', type=str, help='Text to speak.', default="I am a language model that has learned to speak.")
    parser.add_argument('-voice', type=str, help='Use a preset conditioning voice (defined above). Overrides cond_path.', default='dotrice,harris,lescault,otto,atkins,grace,kennard,mol')
    parser.add_argument('-num_samples', type=int, help='How many total outputs the autoregressive transformer should produce.', default=512)
    parser.add_argument('-num_batches', type=int, help='How many batches those samples should be produced over.', default=16)
    parser.add_argument('-num_outputs', type=int, help='Number of outputs to produce.', default=2)
    parser.add_argument('-output_path', type=str, help='Where to store outputs.', default='results/')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    download_models()

    for voice in args.voice.split(','):
        print("Loading data..")
        tokenizer = VoiceBpeTokenizer()
        text = torch.IntTensor(tokenizer.encode(args.text)).unsqueeze(0).cuda()
        text = F.pad(text, (0,1))  # This may not be necessary.
        cond_paths = preselected_cond_voices[voice]
        conds = []
        for cond_path in cond_paths:
            c, cond_wav = load_conditioning(cond_path)
            conds.append(c)
        conds = torch.stack(conds, dim=1)
        cond_diffusion = cond_wav[:, :88200]  # The diffusion model expects <= 88200 conditioning samples.

        print("Loading GPT TTS..")
        autoregressive = UnifiedVoice(max_mel_tokens=300, max_text_tokens=200, max_conditioning_inputs=2, layers=30, model_dim=1024,
                                      heads=16, number_text_tokens=256, start_text_token=255, checkpointing=False, train_solo_embeddings=False,
                                      average_conditioning_embeddings=True).cuda().eval()
        autoregressive.load_state_dict(torch.load('.models/autoregressive.pth'))
        stop_mel_token = autoregressive.stop_mel_token

        with torch.no_grad():
            print("Performing autoregressive inference..")
            samples = []
            for b in tqdm(range(args.num_batches)):
                codes = autoregressive.inference_speech(conds, text, num_beams=1, repetition_penalty=1.0, do_sample=True, top_k=50, top_p=.95,
                                                        temperature=.9, num_return_sequences=args.num_samples//args.num_batches, length_penalty=1)
                padding_needed = 250 - codes.shape[1]
                codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
                samples.append(codes)
            del autoregressive

            print("Loading CLIP..")
            clip = VoiceCLIP(dim_text=512, dim_speech=512, dim_latent=512, num_text_tokens=256, text_enc_depth=8, text_seq_len=120, text_heads=8,
                             num_speech_tokens=8192, speech_enc_depth=10, speech_heads=8, speech_seq_len=250).cuda().eval()
            clip.load_state_dict(torch.load('.models/clip.pth'))
            print("Performing CLIP filtering..")
            clip_results = []
            for batch in samples:
                for i in range(batch.shape[0]):
                    batch[i] = fix_autoregressive_output(batch[i], stop_mel_token)
                text = text[:, :120]  # Ugly hack to fix the fact that I didn't train CLIP to handle long enough text.
                clip_results.append(clip(text.repeat(batch.shape[0], 1),
                                    torch.full((batch.shape[0],), fill_value=text.shape[1]-1, dtype=torch.long, device='cuda'),
                                    batch, torch.full((batch.shape[0],), fill_value=batch.shape[1]*1024, dtype=torch.long, device='cuda'),
                                    return_loss=False))
            clip_results = torch.cat(clip_results, dim=0)
            samples = torch.cat(samples, dim=0)
            best_results = samples[torch.topk(clip_results, k=args.num_outputs).indices]

            # Delete the autoregressive and clip models to free up GPU memory
            del samples, clip

            print("Loading Diffusion Model..")
            diffusion = DiffusionTts(model_channels=512, in_channels=100, out_channels=200, in_latent_channels=1024,
                                     channel_mult=[1, 2, 3, 4], num_res_blocks=[3, 3, 3, 3], token_conditioning_resolutions=[1,4,8],
                                     dropout=0, attention_resolutions=[4,8], num_heads=8, kernel_size=3, scale_factor=2,
                                     time_embed_dim_multiplier=4, unconditioned_percentage=0, conditioning_dim_factor=2,
                                     conditioning_expansion=1)
            diffusion.load_state_dict(torch.load('.models/diffusion.pth'))
            diffusion = diffusion.cuda().eval()
            print("Loading vocoder..")
            vocoder = UnivNetGenerator()
            vocoder.load_state_dict(torch.load('.models/vocoder.pth')['model_g'])
            vocoder = vocoder.cuda()
            vocoder.eval(inference=True)
            diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=100)

            print("Performing vocoding..")
            # Perform vocoding on each batch element separately: The diffusion model is very memory (and compute!) intensive.
            for b in range(best_results.shape[0]):
                code = best_results[b].unsqueeze(0)
                mel = do_spectrogram_diffusion(diffusion, diffuser, code, cond_diffusion, mean=False)
                wav = vocoder.inference(mel)
                torchaudio.save(os.path.join(args.output_path, f'{voice}_{b}.wav'), wav.squeeze(0).cpu(), 24000)
