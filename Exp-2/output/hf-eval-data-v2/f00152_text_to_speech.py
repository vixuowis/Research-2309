# function_import --------------------

from transformers import pipeline

# function_code --------------------

def text_to_speech(text: str) -> str:
    """
    Converts a given text into spoken instructions using a Text-to-Speech model.

    Args:
        text (str): The text to be converted into speech.

    Returns:
        str: The spoken instructions.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    tts_pipeline = pipeline('text-to-speech', model='espnet/kan-bayashi_ljspeech_vits')
    spoken_instructions = tts_pipeline(text)
    return spoken_instructions

# test_function_code --------------------

def test_text_to_speech():
    """
    Tests the text_to_speech function by providing a sample text and checking the type of the output.
    """
    sample_text = 'Example instruction for the visually impaired user.'
    output = text_to_speech(sample_text)
    assert isinstance(output, str), 'Output must be a string.'

# call_test_function_code --------------------

test_text_to_speech()