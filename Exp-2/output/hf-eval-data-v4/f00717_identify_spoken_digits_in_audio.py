# requirements_file --------------------

!pip install -U transformers==4.26.1,torch==1.11.0+cpu

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def identify_spoken_digits_in_audio(audio_file_path):
    """
    This function takes the path to an audio file as input,
    and uses a pre-trained model to identify spoken digits in the audio.

    Parameters:
        audio_file_path (str): The path to the audio file to be analyzed.

    Returns:
        list: A list of identified digits in the order they were mentioned in the audio.
    """
    # Load the pre-trained model for spoken digit classification
    digit_classifier = pipeline('audio-classification', model='mazkooleg/0-9up-data2vec-audio-base-960h-ft')

    # Analyze the audio and extract spoken digits
    digits_detected = digit_classifier(audio_file_path)

    # Extract digits from the classifier output
    spoken_digits = [entry['label'] for entry in digits_detected]
    return spoken_digits

# test_function_code --------------------

def test_identify_spoken_digits_in_audio():
    print("Testing started.")

    # Test case 1: Audio file with the spoken digits '123'
    print("Testing case [1/3] started.")
    expected_digits = ['one', 'two', 'three']
    identified_digits = identify_spoken_digits_in_audio('path_to_audio_with_123.wav')
    assert expected_digits == identified_digits, f"Test case [1/3] failed: Expected {expected_digits}, got {identified_digits}"

    # Test case 2: Audio file with no spoken digits
    print("Testing case [2/3] started.")
    identified_digits = identify_spoken_digits_in_audio('path_to_silent_audio.wav')
    assert identified_digits == [], f"Test case [2/3] failed: Expected no digits, got {identified_digits}"

    # Test case 3: Audio file with mixed speech and digits
    print("Testing case [3/3] started.")
    expected_digits = ['zero', 'five', 'nine']
    identified_digits = identify_spoken_digits_in_audio('path_to_audio_with_speech_and_digits.wav')
    assert expected_digits == identified_digits, f"Test case [3/3] failed: Expected {expected_digits}, got {identified_digits}"
    print("Testing finished.")

# Run the test function
test_identify_spoken_digits_in_audio()