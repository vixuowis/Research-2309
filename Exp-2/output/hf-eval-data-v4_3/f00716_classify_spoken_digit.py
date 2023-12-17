# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_spoken_digit(audio_sample_path):
    """
    Classify the digit spoken in an audio sample.

    Args:
        audio_sample_path (str): The file path to the audio sample to classify.

    Returns:
        dict: The classification result containing the predicted digit and confidence.

    Raises:
        ValueError: If audio_sample_path is not a string or is empty.
        OSError: If the audio file cannot be read or does not exist.
    """
    if not isinstance(audio_sample_path, str) or not audio_sample_path:
        raise ValueError('audio_sample_path must be a non-empty string.')
    try:
        spoken_digit_classifier = pipeline('audio-classification', model='MIT/ast-finetuned-speech-commands-v2')
        digit_prediction = spoken_digit_classifier(audio_sample_path)
        return digit_prediction
    except FileNotFoundError:
        raise OSError('The audio file was not found.')

# test_function_code --------------------

def test_classify_spoken_digit():
    print("Testing started.")
    # Placeholder for loading dataset and extracting a sample audio file path
    # dataset = load_dataset("...")
    # sample_audio_path = dataset[0]
    sample_audio_path = 'path/to/test/audio/file.wav'  # Example audio file path

    # Testing case 1: Valid audio file
    print("Testing case [1/2] started.")
    try:
        result = classify_spoken_digit(sample_audio_path)
        assert isinstance(result, dict), f"Test case [1/2] failed: Expected dict, got {type(result)}"
    except Exception as e:
        assert False, f"Test case [1/2] failed: {e}"

    # Testing case 2: Invalid audio file path
    print("Testing case [2/2] started.")
    invalid_path = ''
    try:
        classify_spoken_digit(invalid_path)
        assert False, "Test case [2/2] failed: ValueError not raised for empty path"
    except ValueError as ve:
        assert str(ve) == 'audio_sample_path must be a non-empty string.', f"Test case [2/2] failed: {ve}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_spoken_digit()