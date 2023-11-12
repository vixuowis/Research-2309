# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_spoken_digit(audio_sample_path: str) -> str:
    """
    Classify the spoken digit in the provided audio sample.

    Args:
        audio_sample_path (str): The path to the audio sample file.

    Returns:
        str: The predicted digit.

    Raises:
        OSError: If there is an issue with the file path or the file itself.
    """
    spoken_digit_classifier = pipeline('audio-classification', model='MIT/ast-finetuned-speech-commands-v2')
    digit_prediction = spoken_digit_classifier(audio_sample_path)
    return digit_prediction

# test_function_code --------------------

def test_classify_spoken_digit():
    """
    Test the classify_spoken_digit function with some test cases.
    """
    # Test case 1: An audio sample of the spoken digit '1'
    assert classify_spoken_digit('test_data/spoken_digit_1.wav') == '1'
    # Test case 2: An audio sample of the spoken digit '5'
    assert classify_spoken_digit('test_data/spoken_digit_5.wav') == '5'
    # Test case 3: An audio sample of the spoken digit '9'
    assert classify_spoken_digit('test_data/spoken_digit_9.wav') == '9'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_spoken_digit()