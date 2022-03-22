"""
This model is based on OpenAI's UNet from improved diffusion, with modifications to support a MEL conditioning signal
and an audio conditioning input. It has also been simplified somewhat.
Credit: https://github.com/openai/improved-diffusion
"""
import functools
import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch.nn import Linear
from torch.utils.checkpoint import checkpoint
from x_transformers import ContinuousTransformerWrapper, Encoder

from models.arch_util import normalization, zero_module, Downsample, Upsample, AudioMiniEncoder, AttentionBlock


def is_latent(t):
    return t.dtype == torch.float


def is_sequence(t):
    return t.dtype == torch.long


def ceil_multiple(base, multiple):
    res = base % multiple
    if res == 0:
        return base
    return base + (multiple - res)


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


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        kernel_size=3,
        efficient_config=True,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = {1: 0, 3: 1, 5: 2}[kernel_size]
        eff_kernel = 1 if efficient_config else 3
        eff_padding = 0 if efficient_config else 1

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, eff_kernel, padding=eff_padding),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            Linear(
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
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, eff_kernel, padding=eff_padding)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, x, emb
        )

    def _forward(self, x, emb):
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


class CheckpointedLayer(nn.Module):
    """
    Wraps a module. When forward() is called, passes kwargs that require_grad through torch.checkpoint() and bypasses
    checkpoint for all other args.
    """
    def __init__(self, wrap):
        super().__init__()
        self.wrap = wrap

    def forward(self, x, *args, **kwargs):
        for k, v in kwargs.items():
            assert not (isinstance(v, torch.Tensor) and v.requires_grad)  # This would screw up checkpointing.
        partial = functools.partial(self.wrap, **kwargs)
        return torch.utils.checkpoint.checkpoint(partial, x, *args)


class CheckpointedXTransformerEncoder(nn.Module):
    """
    Wraps a ContinuousTransformerWrapper and applies CheckpointedLayer to each layer and permutes from channels-mid
    to channels-last that XTransformer expects.
    """
    def __init__(self, needs_permute=True, **xtransformer_kwargs):
        super().__init__()
        self.transformer = ContinuousTransformerWrapper(**xtransformer_kwargs)
        self.needs_permute = needs_permute

        for i in range(len(self.transformer.attn_layers.layers)):
            n, b, r = self.transformer.attn_layers.layers[i]
            self.transformer.attn_layers.layers[i] = nn.ModuleList([n, CheckpointedLayer(b), r])

    def forward(self, x, **kwargs):
        if self.needs_permute:
            x = x.permute(0,2,1)
        h = self.transformer(x, **kwargs)
        return h.permute(0,2,1)


class DiffusionTts(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    Customized to be conditioned on an aligned prior derived from a autoregressive
    GPT-style model.

    :param in_channels: channels in the input Tensor.
    :param in_latent_channels: channels from the input latent.
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
            in_latent_channels=1024,
            in_tokens=8193,
            conditioning_dim_factor=8,
            conditioning_expansion=4,
            out_channels=2,  # mean and variance
            dropout=0,
            # res           1, 2, 4, 8,16,32,64,128,256,512, 1K, 2K
            channel_mult=  (1,1.5,2, 3, 4, 6, 8, 12, 16, 24, 32, 48),
            num_res_blocks=(1, 1, 1, 1, 1, 2, 2, 2,   2,  2,  2,  2),
            # spec_cond:    1, 0, 0, 1, 0, 0, 1, 0,   0,  1,  0,  0)
            # attn:         0, 0, 0, 0, 0, 0, 0, 0,   0,  1,  1,  1
            token_conditioning_resolutions=(1,16,),
            attention_resolutions=(512,1024,2048),
            conv_resample=True,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            kernel_size=3,
            scale_factor=2,
            time_embed_dim_multiplier=4,
            freeze_main_net=False,
            efficient_convs=True,  # Uses kernels with width of 1 in several places rather than 3.
            use_scale_shift_norm=True,
            # Parameters for regularization.
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
            # Parameters for super-sampling.
            super_sampling=False,
            super_sampling_max_noising_factor=.1,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if super_sampling:
            in_channels *= 2  # In super-sampling mode, the LR input is concatenated directly onto the input.
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.super_sampling_enabled = super_sampling
        self.super_sampling_max_noising_factor = super_sampling_max_noising_factor
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.alignment_size = 2 ** (len(channel_mult)+1)
        self.freeze_main_net = freeze_main_net
        padding = 1 if kernel_size == 3 else 2
        down_kernel = 1 if efficient_convs else 3

        time_embed_dim = model_channels * time_embed_dim_multiplier
        self.time_embed = nn.Sequential(
            Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            Linear(time_embed_dim, time_embed_dim),
        )

        conditioning_dim = model_channels * conditioning_dim_factor
        # Either code_converter or latent_converter is used, depending on what type of conditioning data is fed.
        # This model is meant to be able to be trained on both for efficiency purposes - it is far less computationally
        # complex to generate tokens, while generating latents will normally mean propagating through a deep autoregressive
        # transformer network.
        self.code_converter = nn.Sequential(
            nn.Embedding(in_tokens, conditioning_dim),
            CheckpointedXTransformerEncoder(
                needs_permute=False,
                max_seq_len=-1,
                use_pos_emb=False,
                attn_layers=Encoder(
                    dim=conditioning_dim,
                    depth=3,
                    heads=num_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_emb_dim=True,
                )
            ))
        self.latent_converter = nn.Conv1d(in_latent_channels, conditioning_dim, 1)
        self.aligned_latent_padding_embedding = nn.Parameter(torch.randn(1,in_latent_channels,1))
        if in_channels > 60:  # It's a spectrogram.
            self.contextual_embedder = nn.Sequential(nn.Conv1d(in_channels,conditioning_dim,3,padding=1,stride=2),
                                                     CheckpointedXTransformerEncoder(
                                                         needs_permute=True,
                                                         max_seq_len=-1,
                                                         use_pos_emb=False,
                                                         attn_layers=Encoder(
                                                             dim=conditioning_dim,
                                                             depth=4,
                                                             heads=num_heads,
                                                             ff_dropout=dropout,
                                                             attn_dropout=dropout,
                                                             use_rmsnorm=True,
                                                             ff_glu=True,
                                                             rotary_emb_dim=True,
                                                         )
                                                     ))
        else:
            self.contextual_embedder = AudioMiniEncoder(1, conditioning_dim, base_channels=32, depth=6, resnet_blocks=1,
                                                        attn_blocks=3, num_attn_heads=8, dropout=dropout, downsample_factor=4, kernel_size=5)
        self.conditioning_conv = nn.Conv1d(conditioning_dim*2, conditioning_dim, 1)
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,conditioning_dim,1))
        self.conditioning_timestep_integrator = TimestepEmbedSequential(
                    ResBlock(conditioning_dim, time_embed_dim, dropout, out_channels=conditioning_dim, kernel_size=1, use_scale_shift_norm=use_scale_shift_norm),
                    AttentionBlock(conditioning_dim, num_heads=num_heads, num_head_channels=num_head_channels),
                    ResBlock(conditioning_dim, time_embed_dim, dropout, out_channels=conditioning_dim, kernel_size=1, use_scale_shift_norm=use_scale_shift_norm),
                    AttentionBlock(conditioning_dim, num_heads=num_heads, num_head_channels=num_head_channels),
                    ResBlock(conditioning_dim, time_embed_dim, dropout, out_channels=conditioning_dim, kernel_size=1, use_scale_shift_norm=use_scale_shift_norm),
        )
        self.conditioning_expansion = conditioning_expansion

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv1d(in_channels, model_channels, kernel_size, padding=padding)
                )
            ]
        )
        token_conditioning_blocks = []
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, (mult, num_blocks) in enumerate(zip(channel_mult, num_res_blocks)):
            if ds in token_conditioning_resolutions:
                token_conditioning_block = nn.Conv1d(conditioning_dim, ch, 1)
                token_conditioning_block.weight.data *= .02
                self.input_blocks.append(token_conditioning_block)
                token_conditioning_blocks.append(token_conditioning_block)

            for _ in range(num_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        kernel_size=kernel_size,
                        efficient_config=efficient_convs,
                        use_scale_shift_norm=use_scale_shift_norm,
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
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(
                            ch, conv_resample, out_channels=out_ch, factor=scale_factor, ksize=down_kernel, pad=0 if down_kernel == 1 else 1
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                kernel_size=kernel_size,
                efficient_config=efficient_convs,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                kernel_size=kernel_size,
                efficient_config=efficient_convs,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, (mult, num_blocks) in list(enumerate(zip(channel_mult, num_res_blocks)))[::-1]:
            for i in range(num_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        kernel_size=kernel_size,
                        efficient_config=efficient_convs,
                        use_scale_shift_norm=use_scale_shift_norm,
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
                        Upsample(ch, conv_resample, out_channels=out_ch, factor=scale_factor)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv1d(model_channels, out_channels, kernel_size, padding=padding)),
        )

    def fix_alignment(self, x, aligned_conditioning):
        """
        The UNet requires that the input <x> is a certain multiple of 2, defined by the UNet depth. Enforce this by
        padding both <x> and <aligned_conditioning> before forward propagation and removing the padding before returning.
        """
        cm = ceil_multiple(x.shape[-1], self.alignment_size)
        if cm != 0:
            pc = (cm-x.shape[-1])/x.shape[-1]
            x = F.pad(x, (0,cm-x.shape[-1]))
            # Also fix aligned_latent, which is aligned to x.
            if is_latent(aligned_conditioning):
                aligned_conditioning = torch.cat([aligned_conditioning,
                                                  self.aligned_latent_padding_embedding.repeat(x.shape[0], 1, int(pc * aligned_conditioning.shape[-1]))], dim=-1)
            else:
                aligned_conditioning = F.pad(aligned_conditioning, (0, int(pc*aligned_conditioning.shape[-1])))
        return x, aligned_conditioning

    def forward(self, x, timesteps, aligned_conditioning, conditioning_input, lr_input=None, conditioning_free=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param aligned_conditioning: an aligned latent or sequence of tokens providing useful data about the sample to be produced.
        :param conditioning_input: a full-resolution audio clip that is used as a reference to the style you want decoded.
        :param lr_input: for super-sampling models, a guidance audio clip at a lower sampling rate.
        :param conditioning_free: When set, all conditioning inputs (including tokens and conditioning_input) will not be considered.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert conditioning_input is not None
        if self.super_sampling_enabled:
            assert lr_input is not None
            if self.training and self.super_sampling_max_noising_factor > 0:
                noising_factor = random.uniform(0,self.super_sampling_max_noising_factor)
                lr_input = torch.randn_like(lr_input) * noising_factor + lr_input
            lr_input = F.interpolate(lr_input, size=(x.shape[-1],), mode='nearest')
            x = torch.cat([x, lr_input], dim=1)

        # Shuffle aligned_latent to BxCxS format
        if is_latent(aligned_conditioning):
            aligned_conditioning = aligned_conditioning.permute(0, 2, 1)

        # Fix input size to the proper multiple of 2 so we don't get alignment errors going down and back up the U-net.
        orig_x_shape = x.shape[-1]
        x, aligned_conditioning = self.fix_alignment(x, aligned_conditioning)

        with autocast(x.device.type, enabled=self.enable_fp16):
            hs = []
            time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

            # Note: this block does not need to repeated on inference, since it is not timestep-dependent.
            if conditioning_free:
                code_emb = self.unconditioned_embedding.repeat(x.shape[0], 1, 1)
            else:
                cond_emb = self.contextual_embedder(conditioning_input)
                if len(cond_emb.shape) == 3:  # Just take the first element.
                    cond_emb = cond_emb[:, :, 0]
                if is_latent(aligned_conditioning):
                    code_emb = self.latent_converter(aligned_conditioning)
                else:
                    code_emb = self.code_converter(aligned_conditioning)
                cond_emb = cond_emb.unsqueeze(-1).repeat(1, 1, code_emb.shape[-1])
                code_emb = self.conditioning_conv(torch.cat([cond_emb, code_emb], dim=1))
            # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
            if self.training and self.unconditioned_percentage > 0:
                unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                                   device=code_emb.device) < self.unconditioned_percentage
                code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(x.shape[0], 1, 1),
                                       code_emb)

            # Everything after this comment is timestep dependent.
            code_emb = torch.repeat_interleave(code_emb, self.conditioning_expansion, dim=-1)
            code_emb = self.conditioning_timestep_integrator(code_emb, time_emb)

            first = True
            time_emb = time_emb.float()
            h = x
            for k, module in enumerate(self.input_blocks):
                if isinstance(module, nn.Conv1d):
                    h_tok = F.interpolate(module(code_emb), size=(h.shape[-1]), mode='nearest')
                    h = h + h_tok
                else:
                    with autocast(x.device.type, enabled=self.enable_fp16 and not first):
                        # First block has autocast disabled to allow a high precision signal to be properly vectorized.
                        h = module(h, time_emb)
                    hs.append(h)
                first = False
            h = self.middle_block(h, time_emb)
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, time_emb)

        # Last block also has autocast disabled for high-precision outputs.
        h = h.float()
        out = self.out(h)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        params = [self.aligned_latent_padding_embedding, self.unconditioned_embedding] + list(self.latent_converter.parameters())
        for p in params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        return out[:, :, :orig_x_shape]


if __name__ == '__main__':
    clip = torch.randn(2, 1, 32868)
    aligned_latent = torch.randn(2,388,1024)
    aligned_sequence = torch.randint(0,8192,(2,388))
    cond = torch.randn(2, 1, 44000)
    ts = torch.LongTensor([600, 600])
    model = DiffusionTts(128,
                         channel_mult=[1,1.5,2, 3, 4, 6, 8],
                         num_res_blocks=[2, 2, 2, 2, 2, 2, 1],
                         token_conditioning_resolutions=[1,4,16,64],
                         attention_resolutions=[],
                         num_heads=8,
                         kernel_size=3,
                         scale_factor=2,
                         time_embed_dim_multiplier=4,
                         super_sampling=False,
                         efficient_convs=False)
    # Test with latent aligned conditioning
    o = model(clip, ts, aligned_latent, cond)
    # Test with sequence aligned conditioning
    o = model(clip, ts, aligned_sequence, cond)
