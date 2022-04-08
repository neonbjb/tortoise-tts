import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2PreTrainedModel, GPT2Config
from models.xtransformers import TransformerWrapper, Encoder, Decoder
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from models.arch_util import AttentionBlock


class InferenceModel(GPT2PreTrainedModel):
    """
    Implementation of GPT2PreTrainedModel from transformers, which allows us to use their generation library with
    this transformer.
    """
    def __init__(self, model):
        super().__init__(GPT2Config())
        self.transformer = model
        self.context = None

    def parallelize(self, device_map=None):
        # Not implemented.
        pass

    def deparallelize(self):
        # Not implemented.
        pass

    def get_output_embeddings(self):
        assert False, "Unsupported operation."

    def set_output_embeddings(self, new_embeddings):
        assert False, "Unsupported operation."

    def store_context(self, context):
        self.context = context

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert self.context is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.transformer.decoder(input_ids, full_context=self.context, return_embeddings=True)
        logits = self.transformer.decoder.to_logits(hidden_states)

        if not return_dict:
            return (logits, )

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None,
            cross_attentions=None,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class ResBlock(nn.Module):
    """
    Basic residual convolutional block that uses GroupNorm.
    """
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan//8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan//8, chan)
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False):
        super().__init__()
        attn = []
        self.init = nn.Sequential(nn.Conv1d(spec_dim, embedding_dim//4, kernel_size=5, padding=2),
                                  nn.Conv1d(embedding_dim//4, embedding_dim//2, kernel_size=3, padding=1, stride=2),
                                  ResBlock(embedding_dim//2),
                                  nn.Conv1d(embedding_dim//2, embedding_dim, kernel_size=3, padding=1, stride=2))
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, do_checkpoint=do_checkpointing))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        return h.mean(dim=2)


class AutoregressiveCodegen(nn.Module):
    def __init__(self, model_dim, depth, num_text_tokens=256, num_mel_tokens=8194, dropout=.1):
        super().__init__()
        assert depth >= 8  # This is the minimum bound to support the context interleaving that happens later.

        self.START_TOKEN=8192
        self.STOP_TOKEN=8193
        self.max_text_token_id = num_text_tokens
        self.max_mel_token_id = num_mel_tokens
        self.mel_embedding = ConditioningEncoder(80, model_dim, do_checkpointing=False)
        self.encoder = TransformerWrapper(
                                  num_tokens=num_text_tokens,
                                  use_pos_emb=False,
                                  max_seq_len=-1,
                                  attn_layers = Encoder(
                                      depth=depth,
                                      heads=model_dim//64,
                                      dim=model_dim,
                                      attn_dropout=dropout,
                                      ff_dropout=dropout,
                                      use_rmsnorm=True,
                                      ff_glu=True,
                                      ff_mult=1,
                                      rotary_pos_emb=True,
                                      attn_rel_pos_bias=True,
                                  ))
        self.encoder.norm = nn.Identity()  # This layer and the next are unused.
        self.encoder.to_logits = nn.Identity()
        self.decoder = TransformerWrapper(
                                  num_tokens=num_mel_tokens,
                                  use_pos_emb=False,
                                  max_seq_len=-1,
                                  attn_layers=Decoder(
                                      depth=depth,
                                      heads=model_dim//64,
                                      dim=model_dim,
                                      attn_dropout=dropout,
                                      ff_dropout=dropout,
                                      use_rmsnorm=True,
                                      ff_glu=True,
                                      ff_mult=1,
                                      rotary_pos_emb=True,
                                      cross_attend=True,
                                      attn_rel_pos_bias=True,
                                  ))

    def get_grad_norm_parameter_groups(self):
        return {
            'encoder': list(self.encoder.parameters()),
            'decoder': list(self.decoder.parameters()),
            'minicoder': list(self.mel_embedding.parameters()),
        }

    def forward(self, text_codes, conditioning_signal, mel_codes, wav_lengths, return_loss=True):
        assert text_codes.max() < self.max_text_token_id and text_codes.min() >= 0, f'Invalid text code encountered: {text_codes.max()}, {text_codes.min()}'
        assert mel_codes.max() < self.max_mel_token_id and mel_codes.min() >= 0, f'Invalid mel code encountered: {mel_codes.max()}, {mel_codes.min()}'

        # Format mel_codes with a stop token on the end.
        mel_lengths = wav_lengths // 1024 + 1
        for b in range(mel_codes.shape[0]):
            mel_codes[b, mel_lengths[b]:] = self.STOP_TOKEN
        mel_codes = F.pad(mel_codes, (0, 1), value=self.STOP_TOKEN)

        # Build the context
        if len(conditioning_signal.shape) != 4:
            conditioning_signal = conditioning_signal.unsqueeze(1)
        cond_embs = []
        for i in range(conditioning_signal.shape[1]):
            cond_embs.append(self.mel_embedding(conditioning_signal[:, i]))
        cond_emb = torch.stack(cond_embs, dim=1).mean(dim=1, keepdim=True)
        _, enc_text = self.encoder(text_codes, return_hiddens=True)
        # Interleave cond_emb into the first few contexts.
        full_context = enc_text
        full_context[1] = cond_emb
        full_context[3] = cond_emb
        full_context[6] = cond_emb

        # Execute the decoder
        dec_inputs = F.pad(mel_codes, (1,0), value=self.START_TOKEN)[:, :-1]
        dec = self.decoder(dec_inputs, full_context=full_context)
        if not return_loss:
            return dec
        loss_mel = F.cross_entropy(dec.permute(0,2,1), mel_codes)
        return loss_mel

    def generate(self, conditioning_signal, text_codes, max_tokens=256, **hf_generate_kwargs):
        inference_model = InferenceModel(self)
        # Build the context
        if len(conditioning_signal.shape) != 4:
            conditioning_signal = conditioning_signal.unsqueeze(1)
        cond_embs = []
        for i in range(conditioning_signal.shape[1]):
            cond_embs.append(self.mel_embedding(conditioning_signal[:, i]))
        cond_emb = torch.stack(cond_embs, dim=1).mean(dim=1, keepdim=True)
        _, enc_text = self.encoder(text_codes, return_hiddens=True)
        # Interleave cond_emb into the first few contexts.
        full_context = enc_text
        full_context[1] = cond_emb
        full_context[3] = cond_emb
        full_context[6] = cond_emb
        inference_model.store_context(full_context)

        gen = inference_model.generate(bos_token_id=self.START_TOKEN, pad_token_id=self.STOP_TOKEN, eos_token_id=self.STOP_TOKEN,
                                            max_length=max_tokens, output_attentions=False, return_dict_in_generate=True,
                                            **hf_generate_kwargs)
        return gen.sequences


if __name__ == '__main__':
    codegen = AutoregressiveCodegen(256, 10)
    torch.save(codegen.state_dict(), 'sample.pth')
    #codegen.generate(torch.randn((1,80,120)), torch.randint(0,256,(1,200)))
    codegen(torch.randint(0,256, (2,200)),
            torch.randn(2,80,120),
            torch.randint(0,8192, (2,350)),
            torch.tensor([192,350]))
