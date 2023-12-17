# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_news_comments_sentiment(comments):
    """
    Analyze the sentiment of consumer comments for news articles.

    Parameters:
        comments (list): A list of comments to be analyzed.

    Returns:
        list: Analysis results where each comment is associated with its sentiment label and score.
    """
    sentiment_analyzer = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest')
    return sentiment_analyzer(comments)

# test_function_code --------------------

def test_analyze_news_comments_sentiment():
    print("Testing analyze_news_comments_sentiment function.")
    comments = ['Great article!', 'This is biased.', 'Very informative. Learned a lot!']
    results = analyze_news_comments_sentiment(comments)

    assert len(results) == 3, f"Expected 3 results, but got {len(results)}"
    for result in results:
        assert 'label' in result and 'score' in result, "Each result should contain 'label' and 'score'"
        assert isinstance(result['label'], str), "The 'label' should be a string"
        assert isinstance(result['score'], float), "The 'score' should be a float"

    print("All tests passed for analyze_news_comments_sentiment function.")

# Running the test function
test_analyze_news_comments_sentiment()