# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_article(article_text, model='BaptisteDoyen/camembert-base-xnli', labels=['sport', 'politique', 'santé', 'technologie']):
    """
    Classify the given article text into one of the specified categories.

    Args:
        article_text (str): The text of the article to be classified.
        model (str, optional): The model to be used for classification. Defaults to 'BaptisteDoyen/camembert-base-xnli'.
        labels (List[str], optional): The labels for classification. Defaults to ['sport', 'politique', 'santé', 'technologie'].

    Returns:
        dict: A dictionary with label and score for the article text.

    Raises:
        ValueError: If the article_text is empty or not provided.
    """
    if not article_text:
        raise ValueError('The article_text must be provided and cannot be empty.')

    classifier = pipeline('zero-shot-classification', model=model)
    hypothesis_template = 'Ce texte parle de {}.'
    results = classifier(article_text, labels, hypothesis_template=hypothesis_template)
    return results


# test_function_code --------------------

def test_classify_article():
    print("Testing started.")

    # Test case 1: Correct classification
    print("Testing case [1/3] started.")
    article_text = "L'équipe de France joue aujourd'hui au Parc des Princes"
    classification = classify_article(article_text)
    assert classification['labels'][0] == 'sport', f"Test case [1/3] failed: Expected 'sport', got {classification['labels'][0]}"

    # Test case 2: Model not specified
    print("Testing case [2/3] started.")
    default_model_classification = classify_article(article_text)
    assert default_model_classification == classification, f"Test case [2/3] failed: Default model classification differs"

    # Test case 3: Empty article text
    try:
        print("Testing case [3/3] started.")
        classify_article('')
    except ValueError as e:
        assert str(e) == 'The article_text must be provided and cannot be empty.', f"Test case [3/3] failed: {e}"
    print("Testing finished.")


# call_test_function_line --------------------

test_classify_article()