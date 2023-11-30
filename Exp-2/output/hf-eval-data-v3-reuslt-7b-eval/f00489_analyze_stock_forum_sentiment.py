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

    # Initialize tokenizer and classifier.
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('./sentiment/', return_dict=True)

    # Pre-process text for classification.
    classifier = pipeline('sentiment-analysis')
    forum_posts = [classifier(x, truncation=True)[0]['label'] for x in forum_posts]

    results = []
    for i in range(len(forum_posts)):
        # Tokenize the text and prepare it as a tensor.
        input_ids = tokenizer.encode(forum_posts[i], return_tensors="pt") 
        
        # Get the model prediction.
        result = model(input_ids)[0]

        if 'negative' in str(result):
            results.append('negative')
        else:
            results.append('positive')
            
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