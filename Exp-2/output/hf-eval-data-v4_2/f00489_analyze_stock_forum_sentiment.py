# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline
import pandas as pd

# function_code --------------------

def analyze_stock_forum_sentiment(forum_posts):
    """Analyze the sentiment of stock forum posts using a pre-trained model.

    Args:
        forum_posts (pd.Series): A pandas series containing the forum posts to analyze.

    Returns:
        list: A list of dictionaries with the sentiment analysis results.

    Raises:
        ValueError: If the forum_posts are not in a pd.Series.
    """
    if not isinstance(forum_posts, pd.Series):
        raise ValueError('forum_posts must be a pd.Series')
    tokenizer_loaded = RobertaTokenizer.from_pretrained('zhayunduo/roberta-base-stocktwits-finetuned')
    model_loaded = RobertaForSequenceClassification.from_pretrained('zhayunduo/roberta-base-stocktwits-finetuned')
    nlp = pipeline('text-classification', model=model_loaded, tokenizer=tokenizer_loaded)
    results = nlp(list(forum_posts))
    return results

# test_function_code --------------------

def test_analyze_stock_forum_sentiment():
    print("Testing started.")
    # Create sample data for testing
    forum_posts = pd.Series([
        "Stock X is going up!",
        "I'm selling my shares.",
        "Buy now before it's too late!"
    ])

    # Testing case 1: Check if the function returns a list
    print("Testing case [1/3] started.")
    results = analyze_stock_forum_sentiment(forum_posts)
    assert isinstance(results, list), f"Test case [1/3] failed: Expected a list, got {type(results)}"

    # Testing case 2: Check if each item in the list is a dict
    print("Testing case [2/3] started.")
    for result in results:
        assert isinstance(result, dict), f"Test case [2/3] failed: Expected a dict, got {type(result)}"

    # Testing case 3: Check if ValueError is raised when input is not a pd.Series
    print("Testing case [3/3] started.")
    non_series_input = ["Invalid input type"]
    try:
        analyze_stock_forum_sentiment(non_series_input)
        assert False, "Test case [3/3] failed: ValueError was not raised"
    except ValueError as e:
        assert str(e) == "forum_posts must be a pd.Series", f"Test case [3/3] failed: {str(e)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_stock_forum_sentiment()