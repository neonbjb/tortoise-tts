import os
from random import shuffle

import torchaudio

from api import TextToSpeech
from utils.audio import load_audio


def permutations(args):
    res = []
    k = next(iter(args.keys()))
    vals = args[k]
    del args[k]
    if not args:
        return [{k: v} for v in vals]
    lower = permutations(args)
    for v in vals:
        for l in lower:
            lc = l.copy()
            lc[k] = v
            res.append(lc)
    return res


if __name__ == '__main__':
    fname = 'Y:\\libritts\\test-clean\\transcribed-brief-w2v.tsv'
    outpath_base = 'D:\\tmp\\tortoise-tts-eval\\std_sweep_diffusion'
    outpath_real = 'D:\\tmp\\tortoise-tts-eval\\real'

    arg_ranges = {
        'diffusion_temperature': [.5, .7, 1],
        'cond_free_k': [.5, 1, 2],
    }
    cfgs = permutations(arg_ranges)
    shuffle(cfgs)

    for cfg in cfgs:
        outpath = os.path.join(outpath_base, f'{cfg["cond_free_k"]}_{cfg["diffusion_temperature"]}')
        os.makedirs(outpath, exist_ok=True)
        os.makedirs(outpath_real, exist_ok=True)
        with open(fname, 'r', encoding='utf-8') as f:
            lines = [l.strip().split('\t') for l in f.readlines()]

        recorder = open(os.path.join(outpath, 'transcript.tsv'), 'w', encoding='utf-8')
        tts = TextToSpeech()
        for e, line in enumerate(lines):
            transcript = line[0]
            if len(transcript) > 120:
                continue  # We need to support this, but cannot yet.
            path = os.path.join(os.path.dirname(fname), line[1])
            cond_audio = load_audio(path, 22050)
            torchaudio.save(os.path.join(outpath_real, os.path.basename(line[1])), cond_audio, 22050)
            sample = tts.tts(transcript, [cond_audio, cond_audio], num_autoregressive_samples=256, k=1, diffusion_iterations=200, cond_free=False,
                             repetition_penalty=1.5, length_penalty=2, temperature=.9, top_p=.9)
            down = torchaudio.functional.resample(sample, 24000, 22050)
            fout_path = os.path.join(outpath, os.path.basename(line[1]))
            torchaudio.save(fout_path, down.squeeze(0), 22050)
            recorder.write(f'{transcript}\t{fout_path}\n')
            recorder.flush()
        recorder.close()