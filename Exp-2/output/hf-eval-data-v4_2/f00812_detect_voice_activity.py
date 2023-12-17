# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_voice_activity(audio_file_path: str):
    """Detects voice activity in an audio file.

    Args:
        audio_file_path: The file path of the audio file to analyze.

    Returns:
        A list of dicts representing the moments of voice activity.

    Raises:
        FileNotFoundError: If the audio file is not found at the given path.
        RuntimeError: If the voice activity detection model fails.
    """
    # Create a voice activity detection model
    voice_activity_detector = pipeline('voice-activity-detection', model='funasr/FSMN-VAD')
    
    # Perform voice activity detection
    try:
        voice_activity = voice_activity_detector(audio_file_path)
        return voice_activity
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Audio file not found: {audio_file_path}')
    except Exception as e:
        raise RuntimeError('Voice activity detection failed') from e

# test_function_code --------------------

def test_detect_voice_activity():
    print("Testing started.")

    # Test case 1: Valid audio file
    print("Testing case [1/3] started.")
    valid_file_path = 'test_files/valid_audio.wav'
    assert detect_voice_activity(valid_file_path) is not None, "Test case [1/3] failed: Expected voice activity detected, got None"

    # Test case 2: Non-existing audio file
    print("Testing case [2/3] started.")
    invalid_file_path = 'test_files/non_existing_audio.wav'
    try:
        detect_voice_activity(invalid_file_path)
        assert False, "Test case [2/3] failed: Expected FileNotFoundError, no exception raised"
    except FileNotFoundError:
        pass

    # Test case 3: Invalid file format
    print("Testing case [3/3] started.")
    invalid_format_path = 'test_files/invalid_format.txt'
    try:
        detect_voice_activity(invalid_format_path)
        assert False, "Test case [3/3] failed: Expected RuntimeError, no exception raised"
    except RuntimeError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_detect_voice_activity()