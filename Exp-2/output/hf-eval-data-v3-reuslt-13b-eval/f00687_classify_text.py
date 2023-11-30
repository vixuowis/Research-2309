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

    model = AutoModelForSequenceClassification.from_pretrained(
        "valhalla/distilbart-mnli-12-3"
    )
    tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-3")
    inputs = tokenizer(sequence, return_tensors="pt")
    output = model(**inputs)["logits"][0]
    result = torch.softmax(output, 0).tolist()
    sorted_result = [(candidate_labels[i], result[i]) for i in range(len(result))]
    sorted_result.sort(key=lambda x: x[1], reverse=True)
    return sorted_result[0][0]

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