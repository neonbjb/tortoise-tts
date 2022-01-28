import argparse
import os
import random

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm import tqdm

from models.arch_util import TorchMelSpectrogram
from models.discrete_diffusion_vocoder import DiscreteDiffusionVocoder
from models.lucidrains_dvae import DiscreteVAE
from models.text_voice_clip import VoiceCLIP
from models.unified_voice import UnifiedVoice
from utils.audio import load_audio
from utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from utils.tokenizer import VoiceBpeTokenizer


def load_discrete_vocoder_diffuser(trained_diffusion_steps=4000, desired_diffusion_steps=200):
    """
    Helper function to load a GaussianDiffusion instance configured for use as a vocoder.
    """
    return SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps))


def do_spectrogram_diffusion(diffusion_model, dvae_model, diffuser, mel_codes, conditioning_input, spectrogram_compression_factor=128):
    """
    Uses the specified diffusion model and DVAE model to convert the provided MEL & conditioning inputs into an audio clip.
    """
    with torch.no_grad():
        mel = dvae_model.decode(mel_codes)[0]

        # Pad MEL to multiples of 2048//spectrogram_compression_factor
        msl = mel.shape[-1]
        dsl = 2048 // spectrogram_compression_factor
        gap = dsl - (msl % dsl)
        if gap > 0:
            mel = torch.nn.functional.pad(mel, (0, gap))

        output_shape = (mel.shape[0], 1, mel.shape[-1] * spectrogram_compression_factor)
        return diffuser.p_sample_loop(diffusion_model, output_shape, model_kwargs={'spectrogram': mel, 'conditioning_input': conditioning_input})


def load_conditioning(path, sample_rate=22050, cond_length=44100):
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


if __name__ == '__main__':
    preselected_cond_voices = {
        'simmons': ['Y:\\clips\\books1\\754_Dan Simmons - The Rise Of Endymion 356 of 450\\00026.wav'],
        'news_girl': ['Y:\\clips\\podcasts-0\\8288_20210113-Is More Violence Coming_\\00022.wav', 'Y:\\clips\\podcasts-0\\8288_20210113-Is More Violence Coming_\\00016.wav'],
        'dan_carlin': ['Y:\\clips\\books1\\5_dchha06 Shield of the West\\00476.wav', 'Y:\\clips\\books1\\15_dchha16 Nazi Tidbits\\00036.wav'],
        'libri_test': ['Y:\\libritts\\test-clean\\672\\122797\\672_122797_000057_000002.wav'],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-autoregressive_model_path', type=str, help='Autoregressive model checkpoint to load.', default='.models/unified_voice.pth')
    parser.add_argument('-clip_model_path', type=str, help='CLIP model checkpoint to load.', default='.models/clip.pth')
    parser.add_argument('-diffusion_model_path', type=str, help='Diffusion model checkpoint to load.', default='./models/diffusion_vocoder.pth')
    parser.add_argument('-dvae_model_path', type=str, help='DVAE model checkpoint to load.', default='./models/dvae.pth')
    parser.add_argument('-text', type=str, help='Text to speak.', default="I am a language model that has learned to speak.")
    parser.add_argument('-cond_preset', type=str, help='Use a preset conditioning voice (defined above). Overrides cond_path.', default='dan_carlin')
    parser.add_argument('-num_samples', type=int, help='How many total outputs the autoregressive transformer should produce.', default=32)
    parser.add_argument('-num_batches', type=int, help='How many batches those samples should be produced over.', default=2)
    parser.add_argument('-num_outputs', type=int, help='Number of outputs to produce.', default=2)
    parser.add_argument('-output_path', type=str, help='Where to store outputs.', default='results/')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    print("Loading GPT TTS..")
    autoregressive = UnifiedVoice(max_mel_tokens=300, max_text_tokens=200, max_conditioning_inputs=2, layers=30, model_dim=1024, heads=16, number_text_tokens=256, start_text_token=255, checkpointing=False, train_solo_embeddings=False).eval()
    autoregressive.load_state_dict(torch.load(args.autoregressive_model_path))
    stop_mel_token = autoregressive.stop_mel_token

    print("Loading data..")
    tokenizer = VoiceBpeTokenizer()
    text = torch.IntTensor(tokenizer.encode(args.text)).unsqueeze(0).cuda()
    text = F.pad(text, (0,1))  # This may not be necessary.
    cond_paths = preselected_cond_voices[args.cond_preset]
    conds = []
    for cond_path in cond_paths:
        c, cond_wav = load_conditioning(cond_path, cond_length=132300)
        conds.append(c)
    conds = torch.stack(conds, dim=1)  # And just use the last cond_wav for the diffusion model.

    with torch.no_grad():
        print("Performing GPT inference..")
        samples = []
        for b in tqdm(range(args.num_batches)):
            codes = autoregressive.inference_speech(conds, text, num_beams=1, repetition_penalty=1.0, do_sample=True, top_k=50, top_p=.95,
                                                    temperature=.9, num_return_sequences=args.num_samples//args.num_batches, length_penalty=1)
            padding_needed = 250 - codes.shape[1]
            codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
            samples.append(codes)
        samples = torch.cat(samples, dim=0)
        del autoregressive

        print("Loading CLIP..")
        clip = VoiceCLIP(dim_text=512, dim_speech=512, dim_latent=512, num_text_tokens=256, text_enc_depth=8, text_seq_len=120, text_heads=8,
                         num_speech_tokens=8192, speech_enc_depth=10, speech_heads=8, speech_seq_len=250).eval()
        clip.load_state_dict(torch.load(args.clip_model_path))
        print("Performing CLIP filtering..")
        for i in range(samples.shape[0]):
            samples[i] = fix_autoregressive_output(samples[i], stop_mel_token)
        clip_results = clip(text.repeat(samples.shape[0], 1),
                            torch.full((samples.shape[0],), fill_value=text.shape[1]-1, dtype=torch.long, device='cuda'),
                            samples, torch.full((samples.shape[0],), fill_value=samples.shape[1]*1024, dtype=torch.long, device='cuda'),
                            return_loss=False)
        best_results = samples[torch.topk(clip_results, k=args.num_outputs).indices]

        # Delete the autoregressive and clip models to free up GPU memory
        del samples, clip

        print("Loading DVAE..")
        dvae = DiscreteVAE(positional_dims=1, channels=80, hidden_dim=512, num_resnet_blocks=3, codebook_dim=512, num_tokens=8192, num_layers=2,
                           record_codes=True, kernel_size=3, use_transposed_convs=False).eval()
        dvae.load_state_dict(torch.load(args.dvae_model_path))
        print("Loading Diffusion Model..")
        diffusion = DiscreteDiffusionVocoder(model_channels=128, dvae_dim=80, channel_mult=[1, 1, 1.5, 2, 3, 4, 6, 8, 8, 8, 8], num_res_blocks=[1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
                                             spectrogram_conditioning_resolutions=[2,512], attention_resolutions=[512,1024], num_heads=4, kernel_size=3, scale_factor=2,
                                             conditioning_inputs_provided=True, time_embed_dim_multiplier=4).eval()
        diffusion.load_state_dict(torch.load(args.diffusion_model_path))
        diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=100)

        print("Performing vocoding..")
        # Perform vocoding on each batch element separately: Vocoding is very memory (and compute!) intensive.
        for b in range(best_results.shape[0]):
            code = best_results[b].unsqueeze(0)
            wav = do_spectrogram_diffusion(diffusion, dvae, diffuser, code, cond_wav, spectrogram_compression_factor=256)
            torchaudio.save(os.path.join(args.output_path, f'gpt_tts_output_{b}.wav'), wav.squeeze(0).cpu(), 22050)
