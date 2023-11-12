# function_import --------------------

from transformers import pipeline

# function_code --------------------

def sentiment_analysis(message: str) -> dict:
    '''
    This function uses the Hugging Face transformers library to perform sentiment analysis on a given message.
    The model used is 'cardiffnlp/twitter-xlm-roberta-base-sentiment', a multilingual XLM-roBERTa-base model trained on ~198M tweets and finetuned for sentiment analysis.
    
    Args:
        message (str): The message to analyze.
    
    Returns:
        dict: The sentiment analysis result. The keys are 'label' and 'score'. 'label' is the predicted sentiment ('positive', 'negative', or 'neutral'), and 'score' is the confidence score.
    '''
    sentiment_task = pipeline('sentiment-analysis', model='cardiffnlp/twitter-xlm-roberta-base-sentiment')
    return sentiment_task(message)

# test_function_code --------------------

def test_sentiment_analysis():
    '''
    This function tests the sentiment_analysis function with several test cases.
    '''
    # Test case 1: Negative sentiment
    result = sentiment_analysis('I am really frustrated with the service')
    assert result[0]['label'] in ['positive', 'negative', 'neutral']
    
    # Test case 2: Positive sentiment
    result = sentiment_analysis('I am really happy with the service')
    assert result[0]['label'] in ['positive', 'negative', 'neutral']
    
    # Test case 3: Neutral sentiment
    result = sentiment_analysis('The service is okay')
    assert result[0]['label'] in ['positive', 'negative', 'neutral']
    
    return 'All Tests Passed'

# call_test_function_code --------------------

test_sentiment_analysis()