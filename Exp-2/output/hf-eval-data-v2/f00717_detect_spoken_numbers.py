# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_spoken_numbers(audio_file):
    """
    Analyze customer phone call recordings and identify specific numbers mentioned by customers during the call.

    Args:
        audio_file (str): The path to the audio file to be analyzed.

    Returns:
        list: A list of detected digits in the audio file.

    Raises:
        Exception: If the audio file cannot be processed.
    """
    try:
        # Load the 'mazkooleg/0-9up-data2vec-audio-base-960h-ft' model using the pipeline function.
        digit_classifier = pipeline('audio-classification', model='mazkooleg/0-9up-data2vec-audio-base-960h-ft')
        # Use the created digit classifier to analyze the audio from the customer phone call recordings.
        digits_detected = digit_classifier(audio_file)
        return digits_detected
    except Exception as e:
        print(f'Error: {e}')
        raise

# test_function_code --------------------

def test_detect_spoken_numbers():
    """
    Test the function detect_spoken_numbers.
    """
    # Test with a sample audio file.
    # Note: Replace 'sample_audio_file.wav' with the path to a real audio file for this test to work.
    audio_file = 'sample_audio_file.wav'
    detected_digits = detect_spoken_numbers(audio_file)
    # Check if the function returns a list.
    assert isinstance(detected_digits, list), 'The function should return a list.'
    # Check if the list contains integers.
    assert all(isinstance(digit, int) for digit in detected_digits), 'The list should contain integers.'

# call_test_function_code --------------------

test_detect_spoken_numbers()