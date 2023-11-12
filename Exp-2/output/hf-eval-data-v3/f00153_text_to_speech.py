# function_import --------------------

from transformers import pipeline

# function_code --------------------

def text_to_speech(input_text: str) -> bytes:
    """
    Convert input text to speech using ESPnet's Text-to-Speech model.

    Args:
        input_text (str): The text to be converted to speech.

    Returns:
        bytes: The audio data in bytes format.

    Raises:
        OSError: If the specified model is not found.
    """
    try:
        tts = pipeline('text-to-speech', model='mio/Artoria')
        audio = tts(input_text)
        return audio
    except OSError:
        raise OSError('Model not found. Please check the model name.')

# test_function_code --------------------

def test_text_to_speech():
    """
    Test the text_to_speech function with some test cases.
    """
    # Test case 1: Normal case
    audio1 = text_to_speech('This is an example sentence.')
    assert isinstance(audio1, bytes), 'The output type is not correct.'

    # Test case 2: Empty string
    audio2 = text_to_speech('')
    assert isinstance(audio2, bytes), 'The output type is not correct.'

    # Test case 3: Long string
    long_string = 'This is a very long string. ' * 100
    audio3 = text_to_speech(long_string)
    assert isinstance(audio3, bytes), 'The output type is not correct.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_text_to_speech()