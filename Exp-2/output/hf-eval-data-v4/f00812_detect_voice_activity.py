# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_voice_activity(audio_file_path):
    """
    Detect whether there is voice activity in the given audio file.

    Parameters:
    audio_file_path (str): The path to the audio file to analyze.

    Returns:
    dict: The result of voice activity detection, indicating the times when the user is speaking.
    """
    # Load the voice activity detection model
    voice_activity_detector = pipeline('voice-activity-detection', model='funasr/FSMN-VAD')

    # Perform voice activity detection
    voice_activity = voice_activity_detector(audio_file_path)

    return voice_activity

# test_function_code --------------------

def test_detect_voice_activity():
    print("Testing detect_voice_activity function.")
    # This is a placeholder for a proper audio file path, which should be replaced with a real path in a proper test.
    sample_audio_file_path = 'path/to/sample_audio.wav'

    # Perform the test
    voice_activity = detect_voice_activity(sample_audio_file_path)
    assert isinstance(voice_activity, dict), "The result must be a dictionary."

    print("Testing completed successfully.")

# Run the test function
test_detect_voice_activity()