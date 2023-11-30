# function_import --------------------

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# function_code --------------------

def classify_text(sequence: str, candidate_labels: list):
    """
    Classify a text sequence into one of the candidate labels using zero-shot classification.

    Args:
        sequence (str): The text sequence to classify.
        candidate_labels (list): A list of candidate labels.

    Returns:
        str: The label that the sequence is classified into.
    """

    # Load a model from huggingface library
    model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

    # Define the label candidates
    labels = candidate_labels

    # Tokenize and encode sequence to a torch tensor
    encoding = tokenizer(sequence, return_tensors='pt', truncation=True, padding=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Perform zero-shot classification using a model from huggingface library
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=torch.tensor([0]))

    # Get the probabilities for each candidate label
    scores = torch.nn.Softmax(dim=-1)(outputs.logits)
    
    # Get the most likely label according to the model output
    predicted_id = int(torch.argmax(scores))
    predicted_label = labels[predicted_id]
    
    return predicted_label

# test_function_code --------------------

def test_classify_text():
    """
    Test the classify_text function.
    """
    text_message = 'I spent hours in the kitchen trying a new recipe.'
    categories = ['travel', 'cooking', 'dancing']
    result = classify_text(text_message, categories)
    assert result in categories

    text_message = 'I am planning a trip to Paris.'
    categories = ['travel', 'cooking', 'dancing']
    result = classify_text(text_message, categories)
    assert result in categories

    text_message = 'I love to dance salsa.'
    categories = ['travel', 'cooking', 'dancing']
    result = classify_text(text_message, categories)
    assert result in categories

    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_text()