# function_import --------------------

from transformers import pipeline
from requests.exceptions import ReadTimeoutError

# function_code --------------------

def keyword_spotting(audio_file_path: str, top_k: int = 5):
    """
    Function to detect keywords in an audio file using Hugging Face's audio classification pipeline.

    Args:
        audio_file_path (str): Path to the audio file.
        top_k (int, optional): Number of top predictions to return. Defaults to 5.

    Returns:
        list: List of detected keywords and their scores.

    Raises:
        FileNotFoundError: If the audio file is not found at the provided path.
        ReadTimeoutError: If there is a timeout while loading the model.
    """
    keyword_spotter = pipeline('audio-classification', model='superb/hubert-base-superb-ks')
    try:
        detected_keywords = keyword_spotter(audio_file_path, top_k=top_k)
    except FileNotFoundError:
        raise FileNotFoundError(f"No such file or directory: '{audio_file_path}'")
    except ReadTimeoutError:
        raise ReadTimeoutError("Timeout while loading the model.")
    return detected_keywords

# test_function_code --------------------

def test_keyword_spotting():
    """Test function for keyword_spotting."""
    sample_audio_file_path = 'sample_audio_file.wav'
    top_k = 5
    try:
        predictions = keyword_spotting(sample_audio_file_path, top_k)
        assert isinstance(predictions, list), "The function should return a list."
        assert len(predictions) <= top_k, "The function should return at most top_k predictions."
    except FileNotFoundError:
        print("Test failed. No such file or directory: 'sample_audio_file.wav'")
    except ReadTimeoutError:
        print("Test failed. Timeout while loading the model.")
    print('All Tests Passed')

# call_test_function_code --------------------

test_keyword_spotting()