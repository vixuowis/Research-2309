# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_emotion(text):
    '''
    Classify the emotion of a given text using the 'joeddav/distilbert-base-uncased-go-emotions-student' model.

    Args:
        text (str): The text to be classified.

    Returns:
        dict: The classified emotion of the text.
    '''
    nlp = pipeline('text-classification', model='joeddav/distilbert-base-uncased-go-emotions-student')
    result = nlp(text)
    return result

# test_function_code --------------------

def test_classify_emotion():
    '''
    Test the classify_emotion function.
    '''
    assert isinstance(classify_emotion('I am so happy today!'), dict)
    assert isinstance(classify_emotion('I am so sad.'), dict)
    assert isinstance(classify_emotion('I am so angry!'), dict)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_emotion()