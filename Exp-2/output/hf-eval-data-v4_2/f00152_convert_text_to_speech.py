# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def convert_text_to_speech(text):
    """Converts input text into spoken instructions using a pre-trained model.

    Args:
        text (str): Text to be converted into speech.

    Returns:
        sound (bytes): The audio bytes representing the spoken instructions.

    Raises:
        ValueError: If the input text is empty or None.
    """
    if not text:
        raise ValueError('Input text cannot be empty or None.')

    tts_pipeline = pipeline('text-to-speech', model='espnet/kan-bayashi_ljspeech_vits')
    sound = tts_pipeline(text)
    return sound

# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing started.")

    # Test case 1: Non-empty input string
    print("Testing case [1/2] started.")
    non_empty_text = 'Hello world!'
    assert convert_text_to_speech(non_empty_text), "Test case [1/2] failed: Expected non-empty audio bytes."

    # Test case 2: Input string is None or empty
    print("Testing case [2/2] started.")
    empty_text = ''
    try:
        convert_text_to_speech(empty_text)
        assert False, "Test case [2/2] failed: Exception expected for empty input."
    except ValueError as e:
        assert str(e) == 'Input text cannot be empty or None.', f"Test case [2/2] failed: {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_convert_text_to_speech()