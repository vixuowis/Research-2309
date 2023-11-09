from transformers import pipeline

def classify_article(sequence_to_classify):
    """
    Classify a given sequence into one of the candidate categories using a zero-shot classification model.

    Args:
        sequence_to_classify (str): The sequence to be classified.

    Returns:
        dict: The classification output which includes the labels and their corresponding scores.
    """
    zero_shot_classifier = pipeline('zero-shot-classification', model='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')
    candidate_labels = ['politics', 'economy', 'entertainment', 'environment']
    classification_output = zero_shot_classifier(sequence_to_classify, candidate_labels, multi_label=False)
    return classification_output