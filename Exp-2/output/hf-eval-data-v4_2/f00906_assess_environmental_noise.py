# requirements_file --------------------

!pip install -U pyannote-audio brouhaha-vad

# function_import --------------------

from pyannote.audio import Model, Inference

# function_code --------------------

def assess_environmental_noise(audio_file_path, access_token):
    """
    Assess the noise levels in the environment using a pre-trained model.

    Args:
        audio_file_path (str): The file path of the audio to be analyzed.
        access_token (str): The access token for Hugging Face Transformers API.

    Returns:
        list: A list of tuples with the time, voice activity detection (VAD), speech-to-noise ratio (SNR),
             and the C50 room acoustics estimation for each frame in the audio file.

    Raises:
        ValueError: If audio_file_path is not provided or empty.
        ValueError: If access_token is not provided or empty.
    """
    if not audio_file_path:
        raise ValueError('The audio_file_path must be provided.')
    if not access_token:
        raise ValueError('An access_token must be provided.')

    model = Model.from_pretrained('pyannote/brouhaha', use_auth_token=access_token)
    inference = Inference(model)
    output = inference(audio_file_path)
    results = []
    for frame, (vad, snr, c50) in output:
        t = frame.middle
        results.append((t, vad, snr, c50))
    return results


# test_function_code --------------------

import os

AUDIO_SAMPLE_PATH = 'audio_test_sample.wav'
ACCESS_TOKEN = 'your_access_token_here'

def test_assess_environmental_noise():
    print("Testing started.")
    
    # Ensuring that the test audio file exists
    if not os.path.exists(AUDIO_SAMPLE_PATH):
        raise FileNotFoundError(f"Audio sample file not found: {AUDIO_SAMPLE_PATH}")

    # Testing the function with a sample audio file
    print("Testing case [1/1] started.")
    results = assess_environmental_noise(AUDIO_SAMPLE_PATH, ACCESS_TOKEN)
    assert isinstance(results, list) and len(results) > 0, "Test case [1/1] failed: Expected a list of non-empty results."
    print("Testing finished.")

# Running the test function
test_assess_environmental_noise()


# call_test_function_line --------------------

test_assess_environmental_noise()