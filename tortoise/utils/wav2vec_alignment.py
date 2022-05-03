import re

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor

from tortoise.utils.audio import load_audio


class Wav2VecAlignment:
    def __init__(self):
        self.model = Wav2Vec2ForCTC.from_pretrained("jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli").cpu()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f"facebook/wav2vec2-large-960h")
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('jbetker/tacotron_symbols')

    def align(self, audio, expected_text, audio_sample_rate=24000, topk=3, return_partial=False):
        orig_len = audio.shape[-1]

        with torch.no_grad():
            self.model = self.model.cuda()
            audio = audio.to('cuda')
            audio = torchaudio.functional.resample(audio, audio_sample_rate, 16000)
            clip_norm = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
            logits = self.model(clip_norm).logits
            self.model = self.model.cpu()

        logits = logits[0]
        w2v_compression = orig_len // logits.shape[0]
        expected_tokens = self.tokenizer.encode(expected_text)
        if len(expected_tokens) == 1:
            return [0]  # The alignment is simple; there is only one token.
        expected_tokens.pop(0)  # The first token is a given.
        next_expected_token = expected_tokens.pop(0)
        alignments = [0]
        for i, logit in enumerate(logits):
            top = logit.topk(topk).indices.tolist()
            if next_expected_token in top:
                alignments.append(i * w2v_compression)
                if len(expected_tokens) > 0:
                    next_expected_token = expected_tokens.pop(0)
                else:
                    break

        if len(expected_tokens) > 0:
            print(f"Alignment did not work. {len(expected_tokens)} were not found, with the following string un-aligned:"
                  f" `{self.tokenizer.decode(expected_tokens)}`. Here's what wav2vec thought it heard:"
                  f"`{self.tokenizer.decode(logits.argmax(-1).tolist())}`")
            if not return_partial:
                return None

        return alignments

    def redact(self, audio, expected_text, audio_sample_rate=24000, topk=3):
        if '[' not in expected_text:
            return audio
        splitted = expected_text.split('[')
        fully_split = [splitted[0]]
        for spl in splitted[1:]:
            assert ']' in spl, 'Every "[" character must be paired with a "]" with no nesting.'
            fully_split.extend(spl.split(']'))
        # Remove any non-alphabetic character in the input text. This makes matching more likely.
        fully_split = [re.sub(r'[^a-zA-Z ]', '', s) for s in fully_split]
        # At this point, fully_split is a list of strings, with every other string being something that should be redacted.
        non_redacted_intervals = []
        last_point = 0
        for i in range(len(fully_split)):
            if i % 2 == 0:
                non_redacted_intervals.append((last_point, last_point + len(fully_split[i]) - 1))
            last_point += len(fully_split[i])

        bare_text = ''.join(fully_split)
        alignments = self.align(audio, bare_text, audio_sample_rate, topk, return_partial=True)
        # If alignment fails, we will attempt to recover by assuming the remaining alignments consume the rest of the string.
        def get_alignment(i):
            if i >= len(alignments):
                return audio.shape[-1]

        output_audio = []
        for nri in non_redacted_intervals:
            start, stop = nri
            output_audio.append(audio[:, get_alignment(start):get_alignment(stop)])
        return torch.cat(output_audio, dim=-1)


if __name__ == '__main__':
    some_audio = load_audio('../../results/train_dotrice_0.wav', 24000)
    aligner = Wav2VecAlignment()
    text = "[God fucking damn it I'm so angry] The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them."
    redact = aligner.redact(some_audio, text)
    torchaudio.save(f'test_output.wav', redact, 24000)
