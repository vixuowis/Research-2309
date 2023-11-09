# function_import --------------------

from transformers import pipeline

# function_code --------------------

def emotion_classification(user_message):
    """
    This function classifies the emotion of a user's message using a pre-trained model.

    Args:
        user_message (str): The user's message that needs to be classified.

    Returns:
        dict: The classified emotion of the user's message.
    """
    emotion_classifier = pipeline('sentiment-analysis', model='michellejieli/emotion_text_classifier')
    emotion_result = emotion_classifier(user_message)
    return emotion_result

# test_function_code --------------------

def test_emotion_classification():
    """
    This function tests the emotion_classification function with a sample user message.
    """
    user_message = 'I am feeling a bit down today.'
    emotion_result = emotion_classification(user_message)
    assert isinstance(emotion_result, list), 'The result should be a list.'
    assert 'label' in emotion_result[0], 'Each item in the list should have a label.'
    assert 'score' in emotion_result[0], 'Each item in the list should have a score.'

# call_test_function_code --------------------

test_emotion_classification()