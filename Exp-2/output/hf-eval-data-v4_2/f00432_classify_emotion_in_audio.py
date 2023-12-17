# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_emotion_in_audio(audio_file_path):
    """Classify the emotion from a German audio speech.

    Args:
        audio_file_path (str): The file path to the German audio speech file.

    Returns:
        dict: A dictionary containing the classification results.

    Raises:
        FileNotFoundError: If the audio file is not found at the specified path.
        Exception: If the classification pipeline fails.
    """
    # Create an audio classifier using the specified model
    audio_classifier = pipeline('audio-classification', model='padmalcom/wav2vec2-large-emotion-detection-german')

    # Check if audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found at {audio_file_path}")

    # Classify the emotion in the audio file
    try:
        result = audio_classifier(audio_file_path)
    except Exception as e:
        raise Exception(f"Error during classification: {e}")

    return result

# test_function_code --------------------

import os
from datasets import load_dataset

def test_classify_emotion_in_audio():
    print("Testing started.")
    # Assume a dataset function that fetches audio speech samples (just for the test scenario)
    dataset = load_dataset("german_speech_emotion")
    sample_data = dataset[0]  # Using the first sample for testing

    example_audio_file_path = sample_data['file_path']

    # Test case 1: Valid audio file path
    print("Testing case [1/2] started.")
    assert isinstance(classify_emotion_in_audio(example_audio_file_path), dict), "Test case [1/2] failed: The result should be a dictionary."

    # Test case 2: Invalid audio file path
    print("Testing case [2/2] started.")
    try:
        classify_emotion_in_audio("non_existing_file.wav")
        assert False, "Test case [2/2] failed: FileNotFoundError was not raised."
    except FileNotFoundError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_emotion_in_audio()