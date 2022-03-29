import os

import torchaudio

from api import TextToSpeech
from utils.audio import load_audio

if __name__ == '__main__':
    fname = 'Y:\\libritts\\test-clean\\transcribed-brief-w2v.tsv'
    outpath = 'D:\\tmp\\tortoise-tts-eval\\redo_outlier'
    outpath_real = 'D:\\tmp\\tortoise-tts-eval\\real'

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
                         top_k=None, top_p=.95, typical_sampling=False, temperature=.7, length_penalty=.5, repetition_penalty=1)
        down = torchaudio.functional.resample(sample, 24000, 22050)
        fout_path = os.path.join(outpath, os.path.basename(line[1]))
        torchaudio.save(fout_path, down.squeeze(0), 22050)
        recorder.write(f'{transcript}\t{fout_path}\n')
        recorder.flush()
    recorder.close()