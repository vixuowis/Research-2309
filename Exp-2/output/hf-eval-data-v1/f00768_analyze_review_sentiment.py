from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch

def analyze_review_sentiment(review_text):
    """
    Analyze the sentiment of a restaurant review using a fine-tuned BERT model.

    Args:
        review_text (str): The text of the restaurant review.

    Returns:
        str: The sentiment of the review ('Positive' or 'Negative').
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    config = AutoConfig.from_pretrained('potatobunny/results-yelp')
    model = AutoModelForSequenceClassification.from_pretrained('potatobunny/results-yelp', config=config)

    # Tokenize the review text and convert to tensor
    inputs = tokenizer(review_text, return_tensors='pt')

    # Predict the sentiment of the review
    outputs = model(**inputs)
    _, prediction = torch.max(outputs.logits, dim=1)

    # Convert the prediction to 'Positive' or 'Negative'
    if prediction.item() == 1:
        return 'Positive'
    else:
        return 'Negative'