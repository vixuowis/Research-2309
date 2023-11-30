# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# function_code --------------------

def analyze_review_sentiment(review_text):
    """
    Analyze the sentiment of a restaurant review using a pre-trained model.

    Args:
        review_text (str): The text of the restaurant review.

    Returns:
        str: The sentiment of the review ('positive' or 'negative').

    Raises:
        OSError: If there is an error loading the pre-trained model or tokenizing the input text.
    """
    
    # Load the pre-trained model and tokenizer.
    try:
        
        config = AutoConfig.from_pretrained("naterush/bert-base-uncased-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained(
            "naterush/bert-base-uncased-sentiment",
            config=config)
        
        tokenizer = AutoTokenizer.from_pretrained("naterush/bert-base-uncased-sentiment")
    except OSError:
        print('There was an error loading the pre-trained model or tokenizing the input text.')
    
    # Prepare the review text for analysis.
    inputs = tokenizer(review_text, return_tensors="pt")
    
    # Analyze the sentiment of the review.
    outputs = model(**inputs)
        
    probabilities = torch.nn.functional.softmax(outputs[0], dim=-1)[0]
    
    positive_probability = round(probabilities[0].item(), 2) * 100
    negative_probability = round(probabilities[1].item(), 2) * 100
        
    if positive_probability > negative_probability:
        sentiment = 'positive'
    else:
        sentiment = 'negative'
    
    # Return the results.
    return sentiment, positive_probability, negative_probability

# test_function_code --------------------

def test_analyze_review_sentiment():
    """
    Test the analyze_review_sentiment function.
    """
    positive_review = 'The food was delicious and the service was excellent.'
    negative_review = 'The food was terrible and the service was poor.'
    assert analyze_review_sentiment(positive_review) == 'positive'
    assert analyze_review_sentiment(negative_review) == 'negative'
    return 'All Tests Passed'


# call_test_function_code --------------------

if __name__ == '__main__':
    print(test_analyze_review_sentiment())