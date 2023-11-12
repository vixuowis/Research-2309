# function_import --------------------

from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline
import pandas as pd

# function_code --------------------

def analyze_stock_forum_sentiment(forum_posts):
    """
    Analyze the sentiment of a stock forum using a pre-trained model.

    Args:
        forum_posts (pd.Series): A pandas Series of forum posts.

    Returns:
        list: A list of sentiment analysis results for each post.
    """
    tokenizer_loaded = RobertaTokenizer.from_pretrained('zhayunduo/roberta-base-stocktwits-finetuned')
    model_loaded = RobertaForSequenceClassification.from_pretrained('zhayunduo/roberta-base-stocktwits-finetuned')
    nlp = pipeline('text-classification', model=model_loaded, tokenizer=tokenizer_loaded)
    results = nlp(list(forum_posts))
    return results

# test_function_code --------------------

def test_analyze_stock_forum_sentiment():
    """
    Test the analyze_stock_forum_sentiment function.
    """
    forum_posts = pd.Series(["Stock X is going up!", "I'm selling my shares.", "Buy now before it's too late!"])
    results = analyze_stock_forum_sentiment(forum_posts)
    assert isinstance(results, list), 'The result should be a list.'
    assert len(results) == len(forum_posts), 'The length of the result should be equal to the length of the input.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_analyze_stock_forum_sentiment()