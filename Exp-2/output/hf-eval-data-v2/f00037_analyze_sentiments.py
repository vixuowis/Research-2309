# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiments(comments):
    """
    Analyze the sentiments of given comments using a pre-trained RoBERTa-base model.

    Args:
        comments (list of str): The comments to be analyzed.

    Returns:
        list of dict: The sentiment analysis results for each comment. Each dict contains the 'label' (either 'POSITIVE' or 'NEGATIVE') and 'score' (a float between 0 and 1).
    """
    sentiment_analyzer = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest')
    return sentiment_analyzer(comments)

# test_function_code --------------------

def test_analyze_sentiments():
    """
    Test the analyze_sentiments function with some example comments.
    """
    comments = ['I love this news!', 'This is terrible.']
    results = analyze_sentiments(comments)
    assert len(results) == len(comments)
    for result in results:
        assert 'label' in result
        assert 'score' in result
        assert result['label'] in ['POSITIVE', 'NEGATIVE']
        assert 0 <= result['score'] <= 1

# call_test_function_code --------------------

test_analyze_sentiments()