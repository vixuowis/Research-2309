# requirements_file --------------------

!pip install -U pyannote.audio

# function_import --------------------

from pyannote.audio.core.inference import Inference

# function_code --------------------

def detect_voice_activity(audio_file_path: str, model_device: str = 'cuda') -> dict:
    """
    Detects voice activity in an audio file using a pre-trained model.

    Args:
        audio_file_path (str): The path to the audio file to be analyzed.
        model_device (str): The device to run the model on, defaults to 'cuda'.

    Returns:
        dict: A dictionary containing the timestamps of detected voice activities.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If there is an issue during the model inference process.
    """
    # Initialize the voice activity detection model
    model = Inference('julien-c/voice-activity-detection', device=model_device)

    # Detect voice activity
    try:
        vads = model({'audio': audio_file_path})
        return vads
    except FileNotFoundError:
        raise FileNotFoundError(f"The audio file {audio_file_path} was not found.")
    except Exception as err:
        raise RuntimeError(f"An error occurred during inference: {err}")


# test_function_code --------------------

import os

# Assumes that 'sample_audio.wav' is available in the current directory
AUDIO_FILE_PATH = 'sample_audio.wav'

# Ensure the audio file exists for testing
assert os.path.exists(AUDIO_FILE_PATH), 'Audio file does not exist for testing.'

# Testing function for detect_voice_activity
def test_detect_voice_activity():
    print("Testing started.")

    # Testing case valid audio file
    print("Testing case [1/2] started.")
    try:
        results = detect_voice_activity(AUDIO_FILE_PATH)
        assert results is not None, 'No results returned from detect_voice_activity function.'
    except Exception as e:
        assert False, f'Test case [1/2] failed: {e}'

    # Testing case invalid audio file
    print("Testing case [2/2] started.")
    try:
        detect_voice_activity('nonexistent_file.wav')
        assert False, 'Test case [2/2] should have raised FileNotFoundError'
    except FileNotFoundError:
        pass
    except Exception as e:
        assert False, f'Test case [2/2] failed: {e}'

    print("Testing finished.")

    return 'Testing completed successfully.'


# call_test_function_line --------------------

test_detect_voice_activity()