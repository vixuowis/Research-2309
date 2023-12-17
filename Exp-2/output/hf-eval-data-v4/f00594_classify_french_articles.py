# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_french_articles(text, categories):
    # This function classifies a French article into predefined categories using a zero-shot classification model.
    # Args:
    #   text (str): The content of the article to classify.
    #   categories (List[str]): A list of categories to classify the article into.
    # Returns:
    #   dict: A dictionary with categories as keys and confidence scores as values.
    classifier = pipeline('zero-shot-classification', model='BaptisteDoyen/camembert-base-xnli')
    hypothesis_template = 'Ce texte parle de {}.'
    return classifier(text, categories, hypothesis_template=hypothesis_template)

# test_function_code --------------------

def test_classify_french_articles():
    print("Testing classify_french_articles function.")
    # Example French article
    article = "L'quipe de France joue aujourd'hui au Parc des Princes"
    categories = ['sport', 'politique', 'sant', 'technologie']

    # Classify the article
    prediction = classify_french_articles(article, categories)
    # Check if the structure of the prediction is correct
    assert isinstance(prediction, dict), "The function should return a dictionary."
    assert set(prediction.keys()).issubset(set(categories)), "The function should return a dictionary with the correct categories."
    # Check if the prediction is reasonable (i.e., 'sport' is the highest scored category for the given article)
    assert prediction.get('sport', 0) > max(prediction.get(key, 0) for key in prediction if key != 'sport'), "The category 'sport' should have the highest score for the given article."
    print("All tests passed!")

test_classify_french_articles()