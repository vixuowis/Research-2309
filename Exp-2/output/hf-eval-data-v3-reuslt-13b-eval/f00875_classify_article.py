# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_article(sequence_to_classify: str, candidate_labels: list) -> dict:
    """
    Classify a given sequence into one of the candidate categories using a zero-shot classification model.

    Args:
        sequence_to_classify (str): The sequence to be classified.
        candidate_labels (list): The list of candidate categories.

    Returns:
        dict: The classification output which includes the label scores.
    """
    
    # Set up a zero-shot classification pipeline using HuggingFace's transformers library.
    pipe = pipeline(task="zero-shot-classification")
    
    # Run the classification model to get prediction results.
    result = pipe(sequence_to_classify, candidate_labels)
    
    return {"label":result["labels"][0], "score":result["scores"][0]}

# test_function_code --------------------

def test_classify_article():
    """Test the classify_article function."""
    sequence_to_classify = 'Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU'
    candidate_labels = ['politics', 'economy', 'entertainment', 'environment']
    classification_output = classify_article(sequence_to_classify, candidate_labels)
    assert isinstance(classification_output, dict)
    assert 'scores' in classification_output
    assert 'labels' in classification_output
    assert len(classification_output['scores']) == len(candidate_labels)
    assert len(classification_output['labels']) == len(candidate_labels)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_article()