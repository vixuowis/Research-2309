# requirements_file --------------------

import subprocess

requirements = ["torch", "transformers", "torchaudio", "datasets"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline
from datasets import load_dataset

# function_code --------------------

def detect_keywords_in_audio(audio_file_path):
    """
    Detects keywords in a short audio clip using a pretrained model.

    Args:
        audio_file_path (str): The file path of the audio clip to analyze.

    Returns:
        list: A list of the most probable keywords detected in the audio clip.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: If the audio classification pipeline encounters an error.
    """
    # Initialize the audio classification pipeline with the specified model
    keyword_classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-ks')
    # Detect keywords in the audio file
    try:
        detected_keywords = keyword_classifier(audio_file_path, top_k=5)
        return detected_keywords
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Audio file not found: {e}")
    except Exception as e:
        raise Exception(f"Error during keyword detection: {e}")

# test_function_code --------------------

def test_detect_keywords_in_audio():
    print("Testing started.")
    dataset = load_dataset('anton-l/superb_demo', 'ks', split='test')
    sample_data = dataset[0]['file']  # Extract a sample audio file from the dataset

    # Test case 1
    print("Testing case [1/1] started.")
    try:
        keywords = detect_keywords_in_audio(sample_data)
        if not isinstance(keywords, list):
            raise AssertionError(f"Test case [1/1] failed: Expected a list, got {type(keywords)}")
    except Exception as e:
        raise AssertionError(f"Test case [1/1] failed: {e}")
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_keywords_in_audio()