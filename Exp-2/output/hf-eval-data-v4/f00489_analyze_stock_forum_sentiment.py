# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline
import pandas as pd

# function_code --------------------

def analyze_stock_forum_sentiment(posts):
    """
    This function analyzes the sentiment of forum posts using a pre-trained sentiment analysis model.
    It uses the Roberta model specifically fine-tuned for stock-related comments on StockTwits.

    Args:
        posts (list of str): A list of strings, where each string is a forum post to be analyzed.

    Returns:
        list of dict: A list of dictionaries that contain the sentiment classification results.
                      Each dictionary will have the keys 'label' and 'score'.
    """
    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('zhayunduo/roberta-base-stocktwits-finetuned')
    model = RobertaForSequenceClassification.from_pretrained('zhayunduo/roberta-base-stocktwits-finetuned')
    
    # Create a classification pipeline
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    
    # Analyze the sentiment for each post
    results = nlp(posts)

    return results

# test_function_code --------------------

def test_analyze_stock_forum_sentiment():
    print("Testing started.")
    
    # Example forum posts
    sample_posts = [
        "Stock X is on the rise, very bullish move!",
        "I think Stock Y is going to plummet, very bearish signal.",
        "Market looks undecided, could go either way."
    ]

    # Expected labels, assuming the model was trained with 'Bullish' or 'Bearish' as labels
    expected_labels = ['LABEL_1', 'LABEL_0', 'LABEL_1']

    # Analyze the sentiment of the sample posts
    print("Testing sentiment analysis functionality started.")
    results = analyze_stock_forum_sentiment(sample_posts)
    
    # Assertions to check if each sentiment is classified correctly
    for i in range(len(sample_posts)):
        assert results[i]['label'] == expected_labels[i], f"Test case failed: Post {i+1} - Expected {expected_labels[i]}, got {results[i]['label']}"

    print("Testing finished.")