# function_import --------------------

from transformers import pipeline

# function_code --------------------

def emotion_classifier(text):
    """
    Identify the type of emotion in a movie review using Hugging Face Transformers.

    Args:
        text (str): The movie review text to classify.

    Returns:
        dict: The predicted emotion and its score.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')
    classifier = pipeline('sentiment-analysis', model='michellejieli/emotion_text_classifier')
    result = classifier(text)
    return result

# test_function_code --------------------

def test_emotion_classifier():
    """
    Test the emotion_classifier function.
    """
    test_text = 'What a fantastic movie! It was so captivating.'
    result = emotion_classifier(test_text)
    assert isinstance(result, list), 'The result should be a list.'
    assert 'label' in result[0], 'Each item in the result should have a label.'
    assert 'score' in result[0], 'Each item in the result should have a score.'

# call_test_function_code --------------------

test_emotion_classifier()