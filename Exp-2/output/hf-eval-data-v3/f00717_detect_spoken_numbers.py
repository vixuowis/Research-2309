# function_import --------------------

from transformers import pipeline
import os

# function_code --------------------

def detect_spoken_numbers(audio_file_path):
    """
    Detects and classifies spoken numbers in an audio file using a pre-trained model.

    Args:
        audio_file_path (str): The path to the audio file to be analyzed.

    Returns:
        list: A list of detected digits in the audio file.

    Raises:
        OSError: If there is an issue with accessing the audio file or the model.
    """
    try:
        # Load the pre-trained model
        digit_classifier = pipeline('audio-classification', model='mazkooleg/0-9up-data2vec-audio-base-960h-ft')
        # Analyze the audio file
        digits_detected = digit_classifier(audio_file_path)
        return digits_detected
    except OSError as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_detect_spoken_numbers():
    """
    Tests the detect_spoken_numbers function with a sample audio file.
    """
    # Test with a sample audio file
    audio_file = 'sample_audio.wav'
    detected_digits = detect_spoken_numbers(audio_file)
    assert isinstance(detected_digits, list), 'The output should be a list.'
    assert all(isinstance(digit, int) for digit in detected_digits), 'All elements in the output list should be integers.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_detect_spoken_numbers()