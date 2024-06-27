import socket
import sounddevice as sd
import numpy as np

def play_audio_stream(client_socket):
    buffer = b''
    stream = sd.OutputStream(samplerate=24000, channels=1, dtype='float32')
    stream.start()

    try:
        while True:
            chunk = client_socket.recv(1024)
            if b"END_OF_AUDIO" in chunk:
                buffer += chunk.replace(b"END_OF_AUDIO", b"")
                if buffer:
                    audio_array = np.frombuffer(buffer, dtype=np.float32)
                    stream.write(audio_array)
                break

            buffer += chunk
            while len(buffer) >= 4096:
                audio_chunk = buffer[:4096]
                audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
                stream.write(audio_array)
                buffer = buffer[4096:]

    finally:
        stream.stop()
        stream.close()

def send_text_to_server(character_name, text, server_ip='localhost', server_port=5000):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))

    try:
        data = f"{character_name}|{text}"
        client_socket.sendall(data.encode('utf-8'))

        play_audio_stream(client_socket)

        print("Audio playback finished.")

    finally:
        client_socket.close()


if __name__ == "__main__":
    character_name ="deniro"
    text = "Hello This is just for a live speaking test"
    send_text_to_server(character_name, text)
