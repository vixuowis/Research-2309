# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_article(sequence_to_classify: str, candidate_labels: list, multi_label: bool = False):
    """
    Classify a given sequence into one of the candidate categories using a zero-shot classification model.

    Args:
        sequence_to_classify (str): The sequence to be classified.
        candidate_labels (list): The list of candidate labels.
        multi_label (bool, optional): Whether to allow multiple labels for the sequence. Defaults to False.

    Returns:
        dict: The classification output which includes the label and score.
    """
    zero_shot_classifier = pipeline('zero-shot-classification', model='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')
    classification_output = zero_shot_classifier(sequence_to_classify, candidate_labels, multi_label=multi_label)
    return classification_output

# test_function_code --------------------

def test_classify_article():
    """
    Test the classify_article function.
    """
    sequence_to_classify = 'Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU'
    candidate_labels = ['politics', 'economy', 'entertainment', 'environment']
    classification_output = classify_article(sequence_to_classify, candidate_labels)
    assert isinstance(classification_output, dict)
    assert 'label' in classification_output
    assert 'score' in classification_output
    assert classification_output['label'] in candidate_labels

# call_test_function_code --------------------

test_classify_article()