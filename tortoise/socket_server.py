import spacy
import threading
import socket
from tortoise.api_fast import TextToSpeech
from utils.audio import load_voices

tts = TextToSpeech()
nlp = spacy.load("en_core_web_sm")


def generate_audio_stream(text, tts, voice_samples):
    print(f"Generating audio stream...: {text}")
    voice_samples, conditioning_latents = load_voices([voice_samples])
    stream = tts.tts_stream(
        text,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        verbose=True,
        stream_chunk_size=40  # Adjust chunk size as needed
    )
    for audio_chunk in stream:
        yield audio_chunk


def split_text(text, max_length=200):
    doc = nlp(text)
    chunks = []
    chunk = []
    length = 0

    for sent in doc.sents:
        sent_length = len(sent.text)
        if length + sent_length > max_length:
            chunks.append(' '.join(chunk))
            chunk = []
            length = 0
        chunk.append(sent.text)
        length += sent_length + 1

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks


def handle_client(client_socket, tts):
    try:
        while True:
            data = client_socket.recv(1024).decode('utf-8')
            if not data:
                break
            character_name, text = data.split('|', 1)
            text_chunks = split_text(text, max_length=200)
            print(text_chunks)
            for chunk in text_chunks:
                audio_stream = generate_audio_stream(chunk, tts, character_name)

                for audio_chunk in audio_stream:
                    audio_data = audio_chunk.cpu().numpy().flatten()
                    client_socket.sendall(audio_data.tobytes())

            client_socket.sendall(b"END_OF_AUDIO")

    finally:
        client_socket.close()
        print("Client disconnected.")


def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 5000))
    server.listen(5)
    print("Server listening on port 5000")

    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_handler = threading.Thread(target=handle_client, args=(client_socket, tts))
        client_handler.start()


if __name__ == "__main__":
    start_server()
