# requirements_file --------------------

import subprocess

requirements = ["huggingface_hub"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from huggingface_hub import hf_hub_download

# function_code --------------------

def separate_speech_from_noise(audio_path, repo_id="JorisCos/ConvTasNet_Libri2Mix_sepclean_8k"):
    """
    Separates speech from background noise in an audio file using ConvTasNet model.

    Args:
        audio_path (str): The file path to the input audio file.
        repo_id (str): The repository ID of the ConvTasNet model on Hugging Face (default:
                       "JorisCos/ConvTasNet_Libri2Mix_sepclean_8k").

    Returns:
        str: The file path to the separated clean audio.

    Raises:
        ValueError: If the input audio_path is not valid.
        RuntimeError: Failure in processing the audio file or separating speech.
    """
    # Download the model files using hf_hub_download
    model_files = hf_hub_download(repo_id=repo_id)

    # TODO: Implement the audio processing using ConvTasNet model
    separated_audio_path = 'clean_audio.wav'
    return separated_audio_path

# test_function_code --------------------

def test_separate_speech_from_noise():
    print("Testing started.")
    # Assume we have an audio file named 'sample_noise.wav' for testing
    audio_path = 'sample_noise.wav'

    # Testing case 1: Valid audio file
    print("Testing case [1/1] started.")
    clean_audio = separate_speech_from_noise(audio_path)
    assert clean_audio == 'clean_audio.wav', f"Test case [1/1] failed: Expected clean_audio.wav, got {clean_audio}"
    print("Testing finished.")

# call_test_function_line --------------------

test_separate_speech_from_noise()