import argparse
import time
from pydub import AudioSegment
from scipy.signal import resample
from tqdm import tqdm
import sounddevice as sd
import soundfile as sf

DURATION = 10.0

def countdown(count: int):
    """Create a countdown for specified number of seconds.

    Args:
        count (int): The amount of milliseconds to wait for.
    """
    for elapsed_ms in range(count):
        print(f'{(count-elapsed_ms)/1000}', end='\r')
        time.sleep(0.001)

def record_until_keypress(
        output_folder: str,
        samplerate: int=22050,
        channels: int=1):
    """Record audio in chunks until interrupted, then chop into 10s fragments.

    Args:
        output_folder (str): Path to the voice folder where the chunks will be saved.
        samplerate (int, optional): Target sample rate. Defaults to 22050.
        channels (int, optional): The number of channels (1=mono, 2=stereo). Defaults to 1.
    """
    recordings = []
    print("·You are recording! Press Ctrl+C to stop.")
    i=0
    try:
        while True:
            recording =sd.rec(
                int(samplerate * DURATION), samplerate=samplerate, channels=channels, blocking=True)
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

def record_audio(
        file_path: str,
        num_samples: int=3,
        samplerate: int=22050,
        channels: int=1,
        timeout: int=5):
    """Record audio with the specified parameters.

    Args:
        file_path (str): Path to the voice folder where the samples will be saved.
        num_samples (int, optional): The amount of samples to save. Defaults to 3.
        samplerate (int, optional): Target sample rate. Defaults to 22050.
        channels (int, optional): The number of channels (1=mono, 2=stereo). Defaults to 1.
        timeout (int, optional): The seconds to wait between samples. Defaults to 5.
    """
    for i in range(num_samples):
        print(f"Preparing to record sample {i+1} in {timeout} seconds...")
        countdown(timeout * 1000)

        print("Recording...")
        recording = sd.rec(int(samplerate * DURATION), samplerate=samplerate,
                           channels=channels, blocking=True)

        fname = f'{file_path}/{i+1}.wav'
        sf.write(fname, recording, samplerate, subtype='FLOAT')

        print(f"Recording of sample {i+1} finished and saved as '{fname}'.")

def chop_audio(input_path: str,
               output_folder: str,
               no_conversion: bool=True,
               samplerate: int=22050):
    """Chop an audio file into chunks with specified duration.

    Args:
        input_path (str): Path to the original voice sample.
        output_folder (str): Path to the voice folder where the sample's chunks will be saved.
        no_conversion (bool, optional): Ignore the conversion to WAV format. Defaults to True.
        samplerate (int, optional): Target sample rate. Defaults to 22050.
    """
    if no_conversion:
        intermediate_path = input_path
    else:
        print("Loading file...")
        audio = AudioSegment.from_file(input_path)

        print("Converting to WAV...")
        intermediate_path = "intermediate.wav"
        audio.export(intermediate_path, format="wav")

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
    parser.add_argument(
        'command',
        choices=['record', 'chop', 'keypress'],
        help='Command to execute')
    parser.add_argument(
        '--file_path',
        help='Path to the file to process')
    parser.add_argument(
        '--output_folder',
        help='Folder to save the chunks')
    parser.add_argument(
        '--no-convert',
        action='store_false', default=True,
        help='Convert file to WAV format')
    parser.add_argument(
        '--num_samples',
        type=int, default=1,
        help='Number of samples to record')
    parser.add_argument(
        '--rec_timeout',
        type=int, default=5,
        help='Seconds between recordings')

    args = parser.parse_args()

    if args.command == 'record':
        record_audio(
            args.file_path,
            num_samples=args.num_samples,
            timeout=args.rec_timeout)
    elif args.command == 'chop':
        chop_audio(
            args.file_path,
            args.output_folder,
            no_conversion=args.no_convert)
    elif args.command == 'keypress':
        record_until_keypress(args.file_path)
