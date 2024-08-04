import subprocess

def local_inference_docker(text, voice, output_path, container_name="tts-app", preset="ultra_fast"):
    """
    Run the TTS Docker container with the specified arguments.

    Args:
    - text (str): The text to be converted to speech.
    - voice (str): The voice to use for the TTS.
    - output_path (str): The path to save the output.
    - container_name (str, optional): The name of the Docker container. Default is "tts-app".
    - preset (str, optional): The preset for the TTS. Default is "ultra_fast".

    Returns:
    - str: Path to the output audio file.
    """
    docker_image = "tts"  # Replace with your Docker image name
    
    # Define the Docker run command
    command = [
        'docker', 'run', '--rm',
        '--name', container_name,
        docker_image,
        '--output_path', output_path,
        '--preset', preset,
        '--voice', voice,
        '--text', text
    ]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(f"Error running Docker container: {result.stderr}")
    return output_path
  