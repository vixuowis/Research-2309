# function_import --------------------

from transformers import pipeline

# function_code --------------------

def keyword_spotting(audio_file_path: str, top_k: int = 5):
    """
    Determine the keyword spoken in a recorded audio file using Hugging Face Transformers.

    Args:
        audio_file_path (str): Path to the audio file.
        top_k (int, optional): Number of top predictions to return. Defaults to 5.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing 'label' and 'score' for each prediction.
    """
    classifier = pipeline('audio-classification', model='superb/hubert-base-superb-ks')
    keyword_predictions = classifier(audio_file_path, top_k=top_k)
    return keyword_predictions

# test_function_code --------------------

def test_keyword_spotting():
    """
    Test the keyword_spotting function.
    """
    # Use a sample audio file for testing
    audio_file_path = 'sample_audio.wav'
    predictions = keyword_spotting(audio_file_path)
    # Check if the function returns the correct number of predictions
    assert len(predictions) == 5
    # Check if each prediction has 'label' and 'score'
    for prediction in predictions:
        assert 'label' in prediction
        assert 'score' in prediction

# call_test_function_code --------------------

test_keyword_spotting()