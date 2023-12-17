# requirements_file --------------------

!pip install -U datasets transformers torchaudio

# function_import --------------------

from datasets import load_dataset
from transformers import pipeline

# function_code --------------------

def detect_keywords_in_audio(audio_file_path, top_k=5):
    """
    Detect the top_k keywords in an audio file.

    :param audio_file_path: str - the path to the audio file.
    :param top_k: int - the number of top keywords to return.
    :return: dict - a dictionary containing detected keywords and their scores.
    """
    # Load the pre-trained keyword spotting model
    classifier = pipeline('audio-classification', model='superb/hubert-base-superb-ks')
    
    # Detect keywords in the audio file
    detected_keywords = classifier(audio_file_path, top_k=top_k)
    
    return detected_keywords

# test_function_code --------------------

def test_detect_keywords_in_audio():
    print("Testing started.")
    # Load a sample audio file for testing
    dataset = load_dataset('anton-l/superb_demo', 'ks', split='test')
    sample_data = dataset[0]['file']

    # Test case: Detect top 5 keywords in the audio
    print("Testing case [1/1] started.")
    predicted_keywords = detect_keywords_in_audio(sample_data, top_k=5)
    assert isinstance(predicted_keywords, dict), "Test case failed: The result should be a dictionary."
    assert len(predicted_keywords) == 5, "Test case failed: The result should contain 5 keywords."
    print("Testing finished.")

# Run the test function
test_detect_keywords_in_audio()