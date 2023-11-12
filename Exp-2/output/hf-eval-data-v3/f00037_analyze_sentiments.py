# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_sentiments(comments):
    """
    Analyze the sentiments of the given comments using a pre-trained RoBERTa-base model.

    Args:
        comments (list): A list of comments to analyze.

    Returns:
        list: A list of dictionaries containing the sentiment analysis results for each comment.
    """
    sentiment_analyzer = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest')
    sentiment_analysis_results = sentiment_analyzer(comments)
    return sentiment_analysis_results

# test_function_code --------------------

def test_analyze_sentiments():
    """
    Test the analyze_sentiments function with some test cases.
    """
    comments = ['I love this news!', 'This is terrible!', 'I am indifferent.']
    results = analyze_sentiments(comments)
    assert len(results) == len(comments)
    for result in results:
        assert result['label'] in ['POSITIVE', 'NEGATIVE']
    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_sentiments()