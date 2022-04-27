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
    fname = 'Y:\\clips\\books2\\subset512-oco.tsv'
    stop_after = 512
    outpath_base = 'D:\\tmp\\tortoise-tts-eval\\sweep-2'
    outpath_real = 'D:\\tmp\\tortoise-tts-eval\\real'

    arg_ranges = {
        'top_p': [.8,1],
        'temperature': [.8,.9,1],
        'diffusion_temperature': [.8,1],
        'cond_free_k': [1,2,5,10],
    }
    cfgs = permutations(arg_ranges)
    shuffle(cfgs)

    for cfg in cfgs:
        cfg_desc = '_'.join([f'{k}-{v}' for k,v in cfg.items()])
        outpath = os.path.join(outpath_base, f'{cfg_desc}')
        os.makedirs(outpath, exist_ok=True)
        os.makedirs(outpath_real, exist_ok=True)
        with open(fname, 'r', encoding='utf-8') as f:
            lines = [l.strip().split('\t') for l in f.readlines()]

        recorder = open(os.path.join(outpath, 'transcript.tsv'), 'w', encoding='utf-8')
        tts = TextToSpeech()
        for e, line in enumerate(lines):
            if e >= stop_after:
                break
            transcript = line[0]
            path = os.path.join(os.path.dirname(fname), line[1])
            cond_audio = load_audio(path, 22050)
            torchaudio.save(os.path.join(outpath_real, os.path.basename(line[1])), cond_audio, 22050)
            sample = tts.tts(transcript, [cond_audio, cond_audio], num_autoregressive_samples=32, repetition_penalty=2.0,
                             k=1, diffusion_iterations=32, length_penalty=1.0, **cfg)
            down = torchaudio.functional.resample(sample, 24000, 22050)
            fout_path = os.path.join(outpath, os.path.basename(line[1]))
            torchaudio.save(fout_path, down.squeeze(0), 22050)
            recorder.write(f'{transcript}\t{fout_path}\n')
            recorder.flush()
        recorder.close()