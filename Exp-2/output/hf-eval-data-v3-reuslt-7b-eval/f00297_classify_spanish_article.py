# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_spanish_article(spanish_article: str, candidate_labels: list, hypothesis_template: str = 'Este ejemplo es {}.') -> dict:
    '''
    Classify a Spanish article into different sections using a pre-trained model.

    Args:
        spanish_article (str): The Spanish article to be classified.
        candidate_labels (list): The list of potential sections the article can be classified into.
        hypothesis_template (str, optional): The template for the classification hypothesis. Defaults to 'Este ejemplo es {}.'.

    Returns:
        dict: The classification results with probabilities for each candidate label.
    '''
    
    # Initialize and load the model
    classifier = pipeline("zero-shot-classification", model="INCEpTION/bartlarge-zeroshot-news")
    
    # Create the inputs to the classifier
    inputs_dict = {
        "candidate_labels": candidate_labels,
        "premises": hypothesis_template.format(spanish_article),
    }
    
    return classifier(**inputs_dict)


# test_function_code --------------------

def test_classify_spanish_article():
    '''
    Test the classify_spanish_article function.
    '''
    spanish_article = 'El autor se perfila, a los 50 años de su muerte, como uno de los grandes de su siglo'
    candidate_labels = ['cultura', 'sociedad', 'economia', 'salud', 'deportes']
    predictions = classify_spanish_article(spanish_article, candidate_labels)
    assert isinstance(predictions, dict), 'The result should be a dictionary.'
    assert 'labels' in predictions, 'The result should contain labels.'
    assert 'scores' in predictions, 'The result should contain scores.'
    assert len(predictions['labels']) == len(candidate_labels), 'The number of labels should be equal to the number of candidate labels.'
    assert len(predictions['scores']) == len(candidate_labels), 'The number of scores should be equal to the number of candidate labels.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_spanish_article()