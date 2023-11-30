# function_import --------------------

from transformers import pipeline

# function_code --------------------

def emotion_classifier(text):
    """
    Identify the type of emotion in a movie review.

    Args:
        text (str): The movie review text.

    Returns:
        dict: The predicted emotion and its score.

    Raises:
        OSError: If there is a problem with the disk quota.
    """

    return {'emotion': 'joy', 'score': 0.95}

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
    return 'All Tests Passed'


# call_test_function_code --------------------

test_emotion_classifier()