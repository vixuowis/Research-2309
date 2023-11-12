# function_import --------------------

from transformers import pipeline

# function_code --------------------

def infer_sentiment(user_comment: str) -> dict:
    '''
    Infer the sentiment of a user comment using zero-shot classification.

    Args:
        user_comment (str): The user comment to infer the sentiment of.

    Returns:
        dict: A dictionary containing the inferred sentiment and the confidence score.
    '''
    nlp = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-6')
    result = nlp(user_comment, ['positive', 'negative'])
    sentiment = result['labels'][0]
    confidence = result['scores'][0]
    return {'sentiment': sentiment, 'confidence': confidence}

# test_function_code --------------------

def test_infer_sentiment():
    '''
    Test the infer_sentiment function.
    '''
    result = infer_sentiment('I recently purchased this product and it completely exceeded my expectations! The build quality is top-notch, and I've already recommended it to several friends.')
    assert isinstance(result, dict)
    assert 'sentiment' in result
    assert 'confidence' in result
    assert result['sentiment'] in ['positive', 'negative']
    assert 0 <= result['confidence'] <= 1

    result = infer_sentiment('I am very disappointed with this product. It broke after just a few days of use.')
    assert isinstance(result, dict)
    assert 'sentiment' in result
    assert 'confidence' in result
    assert result['sentiment'] in ['positive', 'negative']
    assert 0 <= result['confidence'] <= 1

    return 'All Tests Passed'

# call_test_function_code --------------------

test_infer_sentiment()