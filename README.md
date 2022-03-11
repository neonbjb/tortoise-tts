# Tortoise-TTS

Tortoise TTS is an experimental text-to-speech program that uses recent machine learning techniques to generate
high-quality speech samples.

This repo contains all the code needed to run Tortoise TTS in inference mode.

## What's in a name?

I'm naming my speech-related repos after Mojave desert flora and fauna. Tortoise is a bit tongue in cheek: this model
is insanely slow. It leverages both an autoregressive speech alignment model and a diffusion model, both of which
are known for their slow inference. It also performs CLIP sampling, which slows things down even further. You can
expect ~5 seconds of speech to take ~30 seconds to produce on the latest hardware. Still, the results are pretty cool.

## What the heck is this?

Tortoise TTS is inspired by OpenAI's DALLE, applied to speech data. It is made up of 4 separate models that work together.
These models are all derived from different repositories which are all linked. All the models have been modified
for this use case (some substantially so).

First, an autoregressive transformer stack predicts discrete speech "tokens" given a text prompt. This model is very
similar to the GPT model used by DALLE, except it operates on speech data.
Based on: [GPT2 from Transformers](https://huggingface.co/docs/transformers/model_doc/gpt2)

Next, a CLIP model judges a batch of outputs from the autoregressive transformer against the provided text and stack
ranks the outputs according to most probable. You could use greedy or beam-search decoding but in my experience CLIP
decoding creates considerably better results.
Based on [CLIP from lucidrains](https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py)

Next, the speech "tokens" are decoded into a low-quality MEL spectrogram using a VQVAE.
Based on [VQVAE2 by rosinality](https://github.com/rosinality/vq-vae-2-pytorch)

Finally, the output of the VQVAE is further decoded by a UNet diffusion model into raw audio, which can be placed in
a wav file.
Based on [ImprovedDiffusion by openai](https://github.com/openai/improved-diffusion)

## How do I use this?

Check out the colab: https://colab.research.google.com/drive/1wVVqUPqwiDBUVeWWOUNglpGhU3hg_cbR?usp=sharing

Or on a computer with a GPU (with >=16GB of VRAM):
```shell
git clone https://github.com/neonbjb/tortoise-tts.git
cd tortoise-tts
pip install -r requirements.txt
python do_tts.py
```

## Hand-picked TTS samples

I generated ~250 samples from 23 text prompts and 8 voices. The text prompts have never been seen by the model. The
voices were pulled from the training set.

All of the samples can be found in the results/ folder of this repo. I handpicked a few to show what the model is capable of:

-  [Atkins - Road not taken](results/favorites/atkins_road_not_taken.wav)
-  [Dotrice - Rolling Stone interview](results/favorites/dotrice_rollingstone.wav)
-  [Dotrice - 'Ornaments' from tacotron test set](results/favorites/dotrice_tacotron_samp1.wav)
-  [Kennard - 'Acute emotional intelligence' from tacotron test set](results/favorites/kennard_tacotron_samp2.wav)
-  [Mol - Because I could not stop for death](results/favorites/mol_dickenson.wav)
-  [Mol - Obama](results/favorites/mol_obama.wav)

Prosody is remarkably good for poetry, despite the fact that it was never trained on poetry.

## How do I train this?

Frankly - you don't. Building this model has been a labor of love for me, consuming most of my 6 RTX3090s worth of
resources for the better part of 6 months. It uses a dataset I've gathered, refined and transcribed that consists of
a lot of audio data which I cannot distribute because of copywrite or no open licenses.

With that said, I'm willing to help you out if you really want to give it a shot. DM me.

## Looking forward

I'm not satisfied with this yet. Treat this as a "sneak peek" and check back in a couple of months. I think the concept
is sound, but there are a few hurdles to overcome to get sample quality up. I have been doing major tweaks to the
diffusion model and should have something new and much better soon.