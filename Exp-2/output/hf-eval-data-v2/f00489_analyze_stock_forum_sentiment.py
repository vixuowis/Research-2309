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
        A list of dictionaries with the sentiment analysis results for each post.
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
    assert isinstance(results, list), "Result should be a list."
    assert len(results) == len(forum_posts), "Result list length should match input list length."
    for result in results:
        assert 'label' in result, "Each result should have a 'label' key."
        assert 'score' in result, "Each result should have a 'score' key."

# call_test_function_code --------------------

test_analyze_stock_forum_sentiment()