from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

def classify_review_sentiment(review):
    """
    This function classifies the sentiment of a review as either positive or negative.
    It uses the DistilBertForSequenceClassification model from the transformers library.
    
    Parameters:
    review (str): The review to be classified.
    
    Returns:
    str: The sentiment of the review ('POSITIVE' or 'NEGATIVE').
    """
    # Load the tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    
    # Tokenize the input text
    inputs = tokenizer(review, return_tensors='pt')
    
    # Apply the tokenized input to the model and obtain the class logits
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Identify the predicted sentiment class
    predicted_class_id = logits.argmax().item()
    
    # Obtain the corresponding class label
    sentiment = model.config.id2label[predicted_class_id]
    
    return sentiment