# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_german_news(text, candidate_labels):
    """
    Classify German news articles into categories like crime, tragedy, and theft.

    Args:
        text (str): German news article to classify.
        candidate_labels (list): A list of categories to classify the text into.

    Returns:
        dict: The classification result with categories and confidence scores.
    """
    classifier = pipeline('zero-shot-classification', model='Sahajtomar/German_Zeroshot')
    hypothesis_template = 'In diesem Text geht es um {}.'
    result = classifier(text, candidate_labels, hypothesis_template=hypothesis_template)
    return result

# test_function_code --------------------

def test_classify_german_news():
    print("Testing classification of German news articles.")

    # Testing with a predefined German text
    sequence = 'Letzte Woche gab es einen Selbstmord in einer nahe gelegenen Kolonie'
    candidate_labels = ['Verbrechen', 'Tragödie', 'Stehlen']
    result = classify_german_news(sequence, candidate_labels)

    # Check if the output is in the correct format
    assert isinstance(result, dict), "The output should be a dictionary."
    assert 'labels' in result, "The dictionary should have a 'labels' key."
    assert 'scores' in result, "The dictionary should have a 'scores' key."
    assert len(result['labels']) == len(candidate_labels), "The number of labels should match the input."
    assert len(result['scores']) == len(candidate_labels), "The number of scores should match the input."

    print("Test passed successfully!")

# Run the test function
test_classify_german_news()