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
    
    # Load pretrained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(candidate_labels))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Prepare inputs for model (last hidden state)
    encoded_input = tokenizer.encode_plus(sequence, None, return_tensors="pt", add_special_tokens=True)
    input_ids, attention_masks = encoded_input["input_ids"], encoded_input["attention_mask"]

    # Compute prediction
    with torch.no_grad():
        logits = model(input_ids, attention_masks)[0]
    
    # Obtain probability distribution
    probabilities = softmax(logits)

    return candidate_labels[probabilities.argmax()]


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