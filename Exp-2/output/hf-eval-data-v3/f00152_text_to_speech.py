# function_import --------------------

from transformers import pipeline

# function_code --------------------

def text_to_speech(input_text: str) -> str:
    """
    Convert input text to speech using ESPnet's text-to-speech model.

    Args:
        input_text (str): The text to be converted to speech.

    Returns:
        str: The path to the audio file containing the spoken text.

    Raises:
        OSError: If the specified model cannot be found.
    """
    try:
        tts_pipeline = pipeline('text-to-speech', model='espnet/kan-bayashi_ljspeech_vits')
        spoken_instructions = tts_pipeline(input_text)
        return spoken_instructions
    except OSError as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_text_to_speech():
    """
    Test the text_to_speech function with various test cases.
    """
    # Test case 1: Normal case
    assert isinstance(text_to_speech('Hello World'), str)
    # Test case 2: Empty string
    assert isinstance(text_to_speech(''), str)
    # Test case 3: Long string
    assert isinstance(text_to_speech('This is a long string that should be converted to speech.'), str)
    print('All Tests Passed')

# call_test_function_code --------------------

test_text_to_speech()