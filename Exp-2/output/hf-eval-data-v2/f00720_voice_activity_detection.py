# function_import --------------------

from pyannote.audio.core.inference import Inference

# function_code --------------------

def voice_activity_detection(audio_file):
    """
    This function uses the 'julien-c/voice-activity-detection' model from Hugging Face Transformers
    to detect voice activity in an audio file and separate it from silent parts.

    Args:
        audio_file (str): The name of the audio file (e.g. 'TheBigBangTheory.wav').

    Returns:
        result: The result of the voice activity detection. It contains segments of the audio file
        where there is voice activity.

    Raises:
        ValueError: If the audio_file is not a string or if the file does not exist.
    """
    # Check if the audio_file is a string
    if not isinstance(audio_file, str):
        raise ValueError('The audio_file must be a string.')
    # Check if the audio file exists
    if not os.path.isfile(audio_file):
        raise ValueError('The audio file does not exist.')
    # Create an Inference object
    model = Inference('julien-c/voice-activity-detection', device='cuda')
    # Process the audio file
    result = model({
        'audio': audio_file
    })
    return result

# test_function_code --------------------

def test_voice_activity_detection():
    """
    This function tests the voice_activity_detection function by using a sample audio file.
    """
    # Use a sample audio file for testing
    audio_file = 'sample.wav'
    # Call the voice_activity_detection function
    result = voice_activity_detection(audio_file)
    # Check if the result is not None
    assert result is not None, 'The result is None.'
    # Check if the result is a dictionary
    assert isinstance(result, dict), 'The result is not a dictionary.'

# call_test_function_code --------------------

test_voice_activity_detection()