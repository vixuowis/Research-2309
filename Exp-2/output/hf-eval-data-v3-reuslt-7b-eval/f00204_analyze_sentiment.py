# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiment(feedback):
    '''
    Analyze the sentiment of a given text using a pre-trained model from Hugging Face.

    Args:
        feedback (str): The text to be analyzed.

    Returns:
        str: The sentiment of the text, can be 'positive', 'negative', or 'neutral'.
    '''
    if not isinstance(feedback, str):
        raise TypeError('The argument must be a string!')
        
    # Create sentiment analysis object from huggingface
    sentiment_analysis = pipeline("sentiment-analysis")
    
    # Run sentiment analysis on given feedback and return result
    result = sentiment_analysis(feedback)[0]
    if (result['label'] == 'NEGATIVE'): 
        return "negative"
    elif (result['label'] == 'POSITIVE'): 
        return "positive"
    
# function_run --------------------


# test_function_code --------------------

def test_analyze_sentiment():
    '''
    Test the function analyze_sentiment.
    '''
    assert analyze_sentiment('Me encanta este producto!') == 'positive'
    assert analyze_sentiment('No me gusta este producto.') == 'negative'
    assert analyze_sentiment('Este producto es normal.') == 'neutral'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_analyze_sentiment()