"""
This model is based on OpenAI's UNet from improved diffusion, with modifications to support a MEL conditioning signal
and an audio conditioning input. It has also been simplified somewhat.
Credit: https://github.com/openai/improved-diffusion
"""


import math
from abc import abstractmethod

import torch
import torch.nn as nn

from models.arch_util import normalization, zero_module, Downsample, Upsample, AudioMiniEncoder, AttentionBlock


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class TimestepResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        up=False,
        down=False,
        kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = 1 if kernel_size == 3 else (2 if kernel_size == 5 else 0)

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv1d(
                channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class DiscreteSpectrogramConditioningBlock(nn.Module):
    def __init__(self, dvae_channels, channels, level):
        super().__init__()
        self.intg = nn.Sequential(nn.Conv1d(dvae_channels, channels, kernel_size=1),
                                  normalization(channels),
                                  nn.SiLU(),
                                  nn.Conv1d(channels, channels, kernel_size=3))
        self.level = level

    """
    Embeds the given codes and concatenates them onto x. Return shape is the same as x.shape.
    
    :param x: bxcxS waveform latent
    :param codes: bxN discrete codes, N <= S
    """
    def forward(self, x, dvae_in):
        b, c, S = x.shape
        _, q, N = dvae_in.shape
        emb = self.intg(dvae_in)
        emb = nn.functional.interpolate(emb, size=(S,), mode='nearest')
        return torch.cat([x, emb], dim=1)


class DiscreteDiffusionVocoder(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    Customized to be conditioned on a spectrogram prior.

    :param in_channels: channels in the input Tensor.
    :param spectrogram_channels: channels in the conditioning spectrogram.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            model_channels,
            in_channels=1,
            out_channels=2,  # mean and variance
            dvae_dim=512,
            dropout=0,
            # res           1, 2, 4, 8,16,32,64,128,256,512, 1K, 2K
            channel_mult=  (1,1.5,2, 3, 4, 6, 8, 12, 16, 24, 32, 48),
            num_res_blocks=(1, 1, 1, 1, 1, 2, 2, 2,   2,  2,  2,  2),
            # spec_cond:    1, 0, 0, 1, 0, 0, 1, 0,   0,  1,  0,  0)
            # attn:         0, 0, 0, 0, 0, 0, 0, 0,   0,  1,  1,  1
            spectrogram_conditioning_resolutions=(512,),
            attention_resolutions=(512,1024,2048),
            conv_resample=True,
            dims=1,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            kernel_size=3,
            scale_factor=2,
            conditioning_inputs_provided=True,
            time_embed_dim_multiplier=4,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dims = dims

        padding = 1 if kernel_size == 3 else 2

        time_embed_dim = model_channels * time_embed_dim_multiplier
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.conditioning_enabled = conditioning_inputs_provided
        if conditioning_inputs_provided:
            self.contextual_embedder = AudioMiniEncoder(in_channels, time_embed_dim, base_channels=32, depth=6, resnet_blocks=1,
                             attn_blocks=2, num_attn_heads=2, dropout=dropout, downsample_factor=4, kernel_size=5)

        seqlyr = TimestepEmbedSequential(
            nn.Conv1d(in_channels, model_channels, kernel_size, padding=padding)
        )
        seqlyr.level = 0
        self.input_blocks = nn.ModuleList([seqlyr])
        spectrogram_blocks = []
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, (mult, num_blocks) in enumerate(zip(channel_mult, num_res_blocks)):
            if ds in spectrogram_conditioning_resolutions:
                spec_cond_block = DiscreteSpectrogramConditioningBlock(dvae_dim, ch, 2 ** level)
                self.input_blocks.append(spec_cond_block)
                spectrogram_blocks.append(spec_cond_block)
                ch *= 2

            for _ in range(num_blocks):
                layers = [
                    TimestepResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        use_scale_shift_norm=use_scale_shift_norm,
                        kernel_size=kernel_size,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                layer = TimestepEmbedSequential(*layers)
                layer.level = 2 ** level
                self.input_blocks.append(layer)
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                upblk = TimestepEmbedSequential(
                        TimestepResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            kernel_size=kernel_size,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, out_channels=out_ch, factor=scale_factor
                        )
                    )
                upblk.level = 2 ** level
                self.input_blocks.append(upblk)
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            TimestepResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
                kernel_size=kernel_size,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            TimestepResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
                kernel_size=kernel_size,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, (mult, num_blocks) in list(enumerate(zip(channel_mult, num_res_blocks)))[::-1]:
            for i in range(num_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    TimestepResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        use_scale_shift_norm=use_scale_shift_norm,
                        kernel_size=kernel_size,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                        )
                    )
                if level and i == num_blocks:
                    out_ch = ch
                    layers.append(
                        TimestepResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            kernel_size=kernel_size,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channels=out_ch, factor=scale_factor)
                    )
                    ds //= 2
                layer = TimestepEmbedSequential(*layers)
                layer.level = 2 ** level
                self.output_blocks.append(layer)
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv1d(model_channels, out_channels, kernel_size, padding=padding)),
        )

    def forward(self, x, timesteps, spectrogram, conditioning_input=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert x.shape[-1] % 2048 == 0  # This model operates at base//2048 at it's bottom levels, thus this requirement.
        if self.conditioning_enabled:
            assert conditioning_input is not None

        hs = []
        emb1 = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.conditioning_enabled:
            emb2 = self.contextual_embedder(conditioning_input)
            emb = emb1 + emb2
        else:
            emb = emb1

        h = x.type(self.dtype)
        for k, module in enumerate(self.input_blocks):
            if isinstance(module, DiscreteSpectrogramConditioningBlock):
                h = module(h, spectrogram)
            else:
                h = module(h, emb)
                hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


# Test for ~4 second audio clip at 22050Hz
if __name__ == '__main__':
    clip = torch.randn(2, 1, 40960)
    spec = torch.randn(2,80,160)
    cond = torch.randn(2, 1, 40960)
    ts = torch.LongTensor([555, 556])
    model = DiscreteDiffusionVocoder(model_channels=128, channel_mult=[1, 1, 1.5, 2, 3, 4, 6, 8, 8, 8, 8],
                                     num_res_blocks=[1,2, 2, 2, 2, 2, 2, 2, 2,   1,  1 ], spectrogram_conditioning_resolutions=[2,512],
                                     dropout=.05, attention_resolutions=[512,1024], num_heads=4, kernel_size=3, scale_factor=2,
                                     conditioning_inputs_provided=True, conditioning_input_dim=80, time_embed_dim_multiplier=4,
                                     dvae_dim=80)

    print(model(clip, ts, spec, cond).shape)
