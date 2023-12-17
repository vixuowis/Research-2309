# requirements_file --------------------

!pip install -U datasets transformers torchaudio

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_keywords_in_audio(audio_file_path, top_k=5):
    """
    Detects keywords in an audio file using a pre-trained model.

    Args:
        audio_file_path (str): The path to the audio file to be analyzed.
        top_k (int): The number of top probable keywords to return.

    Returns:
        list: A list of keywords detected in the audio file.

    Raises:
        FileNotFoundError: If the audio file cannot be found at the specified path.
    """
    # Initialize the audio classification pipeline with the pre-trained model
    keyword_spotter = pipeline('audio-classification', model='superb/hubert-base-superb-ks')
    # Perform keyword spotting
    detected_keywords = keyword_spotter(audio_file_path, top_k=top_k)
    return detected_keywords

# test_function_code --------------------

from datasets import load_dataset
import os

def test_detect_keywords_in_audio():
    print("Testing started.")
    # Load a sample from the dataset
    dataset = load_dataset('anton-l/superb_demo', 'ks', split='test')
    sample_data = dataset[0]["file"]

    # Ensure the file exists for testing
    assert os.path.isfile(sample_data), "Test sample file does not exist."

    # Testing case 1: The function should return a non-empty list.
    print("Testing case [1/1] started.")
    detected_keywords = detect_keywords_in_audio(sample_data, top_k=5)
    assert detected_keywords, f"Test case [1/1] failed: No keywords detected."
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_keywords_in_audio()