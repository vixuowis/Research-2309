# requirements_file --------------------

import subprocess

requirements = ["soundfile", "asteroid", "huggingface_hub"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import soundfile as sf
from asteroid.models import ConvTasNet
from huggingface_hub import hf_hub_download

# function_code --------------------

def separate_speakers(audio_file_path):
    """Separate speakers from an audio recording using ConvTasNet model.

    Args:
        audio_file_path (str): The file path of the audio recording to process.

    Returns:
        numpy.ndarray: An array containing the separated audio tracks for each speaker.

    Raises:
        ValueError: If the audio file cannot be found or read.
        IOError: If the model cannot be downloaded or loaded properly.
    """
    # Download the pretrained model weights
    model_weights = hf_hub_download(repo_id='JorisCos/ConvTasNet_Libri2Mix_sepclean_8k', filename='model.pth')
    # Load the ConvTasNet model
    model = ConvTasNet.from_pretrained(model_weights)
    # Read the audio file
    mixture_audio, sample_rate = sf.read(audio_file_path)
    # Use the model to separate the speakers
    separated_sources = model.separate(mixture_audio)
    return separated_sources

# test_function_code --------------------

def test_separate_speakers():
    print("Testing started.")
    # Path to a sample audio file (a real file path should be provided in practice)
    audio_file_path = "sample_mixture_audio.wav"
    
    # Test case 1: Check if the function separates the speakers
    print("Testing case [1/1] started.")
    separated_sources = separate_speakers(audio_file_path)
    assert separated_sources is not None and separated_sources.ndim == 2, f"Test case [1/1] failed: Expected a 2D array of separated sources, got {type(separated_sources)} or wrong dimensions ({separated_sources.ndim})"
    print("Testing finished.")

# call_test_function_line --------------------

test_separate_speakers()