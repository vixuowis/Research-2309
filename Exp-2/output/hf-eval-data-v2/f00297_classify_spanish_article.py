# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_spanish_article(spanish_article, candidate_labels):
    """
    Classify a Spanish article into different sections using a pre-trained model.

    Args:
        spanish_article (str): The Spanish article to be classified.
        candidate_labels (list): A list of candidate section labels.

    Returns:
        dict: A dictionary containing the probabilities for each candidate label.
    """
    hypothesis_template = "Este ejemplo es {}."
    classifier = pipeline('zero-shot-classification', model='Recognai/bert-base-spanish-wwm-cased-xnli')
    predictions = classifier(spanish_article, candidate_labels, hypothesis_template=hypothesis_template)
    return predictions

# test_function_code --------------------

def test_classify_spanish_article():
    """
    Test the classify_spanish_article function.
    """
    spanish_article = "El autor se perfila, a los 50 años de su muerte, como uno de los grandes de su siglo"
    candidate_labels = ['cultura', 'sociedad', 'economia', 'salud', 'deportes']
    predictions = classify_spanish_article(spanish_article, candidate_labels)
    assert isinstance(predictions, dict), "The function should return a dictionary."
    assert 'scores' in predictions, "The dictionary should contain a 'scores' key."
    assert len(predictions['scores']) == len(candidate_labels), "The number of scores should be equal to the number of candidate labels."

# call_test_function_code --------------------

test_classify_spanish_article()