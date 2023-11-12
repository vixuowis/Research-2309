# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_keywords_in_audio(audio_file_path, top_k=5):
    """
    Detects keywords in a short audio clip using a pretrained model.

    Args:
        audio_file_path (str): Path to the audio file.
        top_k (int, optional): Number of top probable keywords to return. Defaults to 5.

    Returns:
        list: List of detected keywords and their probabilities.
    """
    keyword_classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-ks')
    detected_keywords = keyword_classifier(audio_file_path, top_k=top_k)
    return detected_keywords

# test_function_code --------------------

def test_detect_keywords_in_audio():
    """
    Tests the detect_keywords_in_audio function with a sample audio file.
    """
    sample_audio_file_path = 'sample_audio.wav'
    detected_keywords = detect_keywords_in_audio(sample_audio_file_path, top_k=5)
    assert isinstance(detected_keywords, list), 'The result is not a list.'
    assert len(detected_keywords) <= 5, 'More than 5 keywords detected.'
    for keyword in detected_keywords:
        assert 'label' in keyword, 'Keyword label is missing.'
        assert 'score' in keyword, 'Keyword score is missing.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_keywords_in_audio()