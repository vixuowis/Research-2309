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
    nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

    probs_list = []
    for label in candidate_labels:
        hypothesis = f'This example is {label}.'
        inputs = tokenizer(sequence, hypothesis, return_tensors='pt', truncation=True)
        logits = nli_model(**inputs)[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1].item()
        probs_list.append(prob_label_is_true)

    category_index = probs_list.index(max(probs_list))
    return candidate_labels[category_index]

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