# requirements_file --------------------

!pip install -U transformers>=4.0.1

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_entities(text):
    """
    Identify names of people and organizations mentioned in a given text using Named Entity Recognition (NER).

    Parameters:
    text (str): The input text to analyze.

    Returns:
    dict: Dictionary with 'people' and 'organizations' lists containing the identified entities.
    """
    nlp = pipeline('ner', model='dslim/bert-large-NER')
    ner_results = nlp(text)

    # Initialize containers for entities
    people = []
    organizations = []
    for entity in ner_results:
        if entity['entity'] == 'B-PER':
            people.append(entity['word'])
        elif entity['entity'] == 'B-ORG':
            organizations.append(entity['word'])
    return {'people': people, 'organizations': organizations}

# test_function_code --------------------

def test_extract_entities():
    print("Testing extract_entities function.")
    sample_text = "I recently purchased a MacBook Pro from Apple Inc. and had a fantastic customer support experience. John from their tech support team was incredibly helpful and professional."
    expected_output = {'people': ['John'], 'organizations': ['Apple Inc']}
    result = extract_entities(sample_text)

    assert result == expected_output, f"Test failed: expected {expected_output}, got {result}"
    print("Test passed: extract_entities function is working correctly.")

# Run the test
test_extract_entities()