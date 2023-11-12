# function_import --------------------

from pyannote.audio.core.inference import Inference

# function_code --------------------

def detect_voice_activity(audio_file_path: str, device: str = 'cuda'):
    """
    Detects voice activity in an audio file using a pre-trained model from Hugging Face Transformers.

    Args:
        audio_file_path (str): Path to the audio file.
        device (str, optional): Device to run the model on. Defaults to 'cuda'.

    Returns:
        dict: A dictionary containing the detected voice activity regions in the audio data.
    """
    model = Inference('julien-c/voice-activity-detection', device=device)
    voice_activity_detection_result = model({'audio': audio_file_path})
    return voice_activity_detection_result

# test_function_code --------------------

def test_detect_voice_activity():
    """
    Tests the detect_voice_activity function with a sample audio file.
    """
    # Test with a sample audio file
    result = detect_voice_activity('sample_audio.wav')
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'audio' in result, 'The result should contain an audio key.'
    assert isinstance(result['audio'], str), 'The audio key should be a string.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_voice_activity()