import argparse
import time
from pydub import AudioSegment
from scipy.signal import resample
from tqdm import tqdm
import sounddevice as sd
import soundfile as sf

DURATION = 10.0


def countdown(count):
    """Create a countdown for specified number of seconds."""
    for s_count in range(count):
        print(f'{(count-s_count)/1000}', end='\r')
        time.sleep(0.001)

def record_until_keypress(output_folder, samplerate=22050, channels=1):
    """Record audio in chunks until interrupted, then chop into 10s fragments."""
    recordings = []
    print("·You are recording! Press Ctrl+C to stop.")
    i=0
    try:
        while True:
            recording = sd.rec(int(samplerate * DURATION), samplerate=samplerate, channels=channels, blocking=True)
            i+=1
            print(f"Chunk #{i} was recorded")
            recordings.append(recording)
    except KeyboardInterrupt:
        print("Recording stopped.")

    for i, rec in enumerate(recordings):
        fname = f'{output_folder}/{i+1}.wav'
        try:
            sf.write(fname, rec, samplerate, subtype='FLOAT')
        except Exception:
            print(f"Error saving chunk #{i+1}.")

def record_audio(file_path, num_samples=1, samplerate=22050, channels=1, timeout=5000):
    """Record audio with the specified parameters."""
    for i in range(num_samples):
        print(f"Preparing to record sample {i+1} in 5 seconds...")
        countdown(timeout)

        print("Recording...")
        recording = sd.rec(int(samplerate * DURATION), samplerate=samplerate,
                           channels=channels, blocking=True)

        fname = f'{file_path}/{i+1}.wav'
        sf.write(fname, recording, samplerate, subtype='FLOAT')

        print(f"Recording of sample {i+1} finished and saved as '{fname}'.")

def chop_audio(input_path, output_folder, convert=False, samplerate=22050):
    """Chop an audio file into chunks with specified duration."""
    if convert:
        print("Loading file...")
        audio = AudioSegment.from_file(input_path)

        print("Converting to WAV...")
        intermediate_path = "intermediate.wav"
        audio.export(intermediate_path, format="wav")
    else:
        intermediate_path = input_path

    print("Loading WAV file...")
    data, og_samplerate = sf.read(intermediate_path)

    print("Resampling audio data...")
    data = resample(data, len(data) * samplerate // og_samplerate)

    num_chunks = len(data) // int(samplerate * DURATION) + 1

    print("Saving chunks...")
    for i, _ in tqdm(enumerate(range(0, len(data), int(samplerate * DURATION)), 1),
                     total=num_chunks):
        chunk = data[i:i + int(samplerate * DURATION)]
        sf.write(f'{output_folder}/{i}.wav', chunk, samplerate, subtype='FLOAT')

    print(f"Conversion and chopping finished. Saved files in '{output_folder}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some audio.')
    parser.add_argument('command', choices=['record', 'chop', 'keypress'], help='Command to execute')
    parser.add_argument('--file_path', help='Path to the file to process')
    parser.add_argument('--output_folder', help='Folder to save the chunks')
    parser.add_argument('--convert', action='store_true', default=False, help='Convert file to WAV format')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to record')
    parser.add_argument('--rec_timeout', type=int, default=5000, help='Milliseconds between recordings')

    args = parser.parse_args()

    if args.command == 'record':
        record_audio(args.file_path, num_samples=args.num_samples, timeout=args.rec_timeout)
    elif args.command == 'chop':
        chop_audio(args.file_path, args.output_folder, convert=args.convert)
    elif args.command == 'keypress':
        record_until_keypress(args.file_path)