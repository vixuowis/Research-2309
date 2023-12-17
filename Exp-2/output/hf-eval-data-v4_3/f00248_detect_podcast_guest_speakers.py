# requirements_file --------------------

import subprocess

requirements = ["pyannote.audio"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model

# function_code --------------------

def detect_podcast_guest_speakers(audio_path: str, access_token: str) -> dict:
    """
    Detects voice activity in a podcast to determine when guests are speaking.

    Args:
        audio_path: str
            The filepath to the podcast audio file.
        access_token: str
            Access token for using Hugging Face models.

    Returns:
        A dictionary with voice activity detection results.

    Raises:
        ValueError: If the audio file is not found or inaccessible.
    """
    # Load pre-trained model from Hugging Face
    model = Model.from_pretrained('pyannote/segmentation', use_auth_token=access_token)
    # Initialize the VoiceActivityDetection pipeline
    pipeline = VoiceActivityDetection(segmentation=model)
    # Define hyperparameters for the VAD pipeline
    HYPER_PARAMETERS = {
        'onset': 0.5, 'offset': 0.5,
        'min_duration_on': 0.0,
        'min_duration_off': 0.0
    }
    # Instantiate the pipeline with the defined hyperparameters
    pipeline.instantiate(HYPER_PARAMETERS)
    # Process the audio file using the VAD pipeline
    vad_result = pipeline(audio_path)
    return vad_result

# test_function_code --------------------

def test_detect_podcast_guest_speakers():
    print("Testing started.")
    audio_sample_path = 'test_audio.wav'  # Replace with a valid audio file path
    access_token = 'test_token'  # Replace with a valid access token

    # Testing case: Check if the function returns a dictionary
    print("Testing case [1/1] started.")
    result = detect_podcast_guest_speakers(audio_sample_path, access_token)
    assert isinstance(result, dict), f"Test case [1/1] failed: Expected result to be a dictionary, but got {type(result)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_podcast_guest_speakers()