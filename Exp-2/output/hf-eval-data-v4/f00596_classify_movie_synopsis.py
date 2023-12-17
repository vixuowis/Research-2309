# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_movie_synopsis(synopsis: str) -> dict:
    """
    Classify a given movie synopsis in German into one of three categories: crime, tragedy, and theft.

    Parameters:
    synopsis (str): The movie synopsis in German to be classified.

    Returns:
    dict: A dictionary containing the classification results.
    """
    # Initialize the zero-shot classification pipeline
    classifier = pipeline('zero-shot-classification', model='Sahajtomar/German_Zeroshot')
    # Candidate labels
    candidate_labels = ['Verbrechen', 'Tragödie', 'Stehlen']
    # German hypothesis template
    hypothesis_template = 'In deisem geht es um {}'
    # Perform classification
    result = classifier(synopsis, candidate_labels, hypothesis_template=hypothesis_template)
    return result


# test_function_code --------------------

def test_classify_movie_synopsis():
    print("Testing classify_movie_synopsis function.")
    # A sample German synopsis
    synopsis = 'Letzte Woche gab es einen Selbstmord in einer nahe gelegenen kolonie'
    result = classify_movie_synopsis(synopsis)
    # Check if the result is a dictionary and has the expected keys
    assert isinstance(result, dict), "The result should be a dictionary."
    assert 'labels' in result, "The result dictionary should contain the key 'labels'."
    assert 'scores' in result, "The result dictionary should contain the key 'scores'."
    assert 'sequence' in result, "The result dictionary should contain the key 'sequence'."
    # Expected label for the provided synopsis
    expected_label = 'Tragödie'
    # Check if the predicted label is in the list of candidate labels
    assert result['labels'][0] == expected_label, f"The predicted label should be {expected_label}."
    print("Testing completed successfully.")
