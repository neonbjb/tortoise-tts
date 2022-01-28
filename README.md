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

Tortoise TTS is inspired by OpenAI's DALLE, applied to speech data. It is made up of 4 separate models that work together:

First, an autoregressive transformer stack predicts discrete speech "tokens" given a text prompt. This model is very
similar to the GPT model used by DALLE, except it operates on speech data.

Next, a CLIP model judges a batch of outputs from the autoregressive transformer against the provided text and stack
ranks the outputs according to most probable. You could use greedy or beam-search decoding but in my experience CLIP
decoding creates considerably better results.

Next, the speech "tokens" are decoded into a low-quality MEL spectrogram using a VQVAE.

Finally, the output of the VQVAE is further decoded by a UNet diffusion model into raw audio, which can be placed in
a wav file.

## How do I use this?

<incoming>

## How do I train this?

Frankly - you don't. Building this model has been a labor of love for me, consuming most of my 6 RTX3090s worth of
resources for the better part of 6 months. It uses a dataset I've gathered, refined and transcribed that consists of
a lot of audio data which I cannot distribute because of copywrite or no open licenses.

With that said, I'm willing to help you out if you really want to give it a shot. DM me.