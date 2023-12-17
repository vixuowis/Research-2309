# requirements_file --------------------

!pip install -U transformers numpy scipy

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_comments_sentiment(comments):
    """
    Analyze the sentiment of consumer comments using a pre-trained sentiment
    analysis model.

    Args:
        comments (list of str): The list of comments to be analyzed.
    
    Returns:
        list of dict: Returns a list of dictionaries with the analysis results
        for each comment, including the label and score.

    Raises:
        ValueError: If `comments` is not a list of strings.
    """
    if not isinstance(comments, list) or not all(isinstance(c, str) for c in comments):
        raise ValueError('The `comments` argument must be a list of strings.')
    # Load the pre-trained sentiment analysis model
    sentiment_analyzer = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest')
    # Perform sentiment analysis
    results = sentiment_analyzer(comments)
    return results

# test_function_code --------------------

def test_analyze_comments_sentiment():
    print('Testing started.')
    # Test case 1: Valid input
    print('Testing case [1/2] started.')
    comments = ['Great news!', 'Terrible article.']
    results = analyze_comments_sentiment(comments)
    assert all('label' in r and 'score' in r for r in results), 'Test case [1/2] failed: Each result should contain a label and a score.'

    # Test case 2: Invalid input
    print('Testing case [2/2] started.')
    non_string_input = [42, None]
    try:
        analyze_comments_sentiment(non_string_input)
        assert False, 'Test case [2/2] failed: ValueError exception was not raised for non-string input.'
    except ValueError as e:
        assert str(e) == 'The `comments` argument must be a list of strings.', 'Test case [2/2] failed: Incorrect error message for non-string input.'

    print('Testing finished.')

# call_test_function_line --------------------

test_analyze_comments_sentiment()