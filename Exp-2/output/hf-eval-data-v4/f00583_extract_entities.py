# requirements_file --------------------

!pip install -U transformers>=4.0.0

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_entities(text):
    """
    Extracts company and person names from a given text using a pre-trained NER model.
    
    Parameters:
    text (str): The input text from which to extract entities.

    Returns:
    list: A list of entities where each entity is a dictionary containing the type ('ORG' or 'PER')
          and the entity text.
    """
    # Load the pre-trained NER model
    ner_model = pipeline('ner', model='Jean-Baptiste/roberta-large-ner-english')

    # Use the NER model to find entities in the text
    ner_results = ner_model(text)

    # Extract and return entities that are either persons or organizations
    entities = [
        {'type': result['entity'], 'text': result['word']}
        for result in ner_results 
        if result['entity'] in ['PER', 'ORG']
    ]
    return entities

# test_function_code --------------------

def test_extract_entities():
    print("Testing started.")
    
    # Test case 1: Sample text with known entities
    print("Testing case [1/3] started.")
    test_text_1 = "Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne."
    expected_result_1 = [
        {'type': 'ORG', 'text': 'Apple'},
        {'type': 'PER', 'text': 'Steve Jobs'},
        {'type': 'PER', 'text': 'Steve Wozniak'},
        {'type': 'PER', 'text': 'Ronald Wayne'}
    ]
    assert extract_entities(test_text_1) == expected_result_1, f"Test case [1/3] failed: Expected {expected_result_1}, got {extract_entities(test_text_1)}"

    # Test case 2: Text without entities should return an empty list
    print("Testing case [2/3] started.")
    test_text_2 = "The quick brown fox jumps over the lazy dog."
    expected_result_2 = []
    assert extract_entities(test_text_2) == expected_result_2, f"Test case [2/3] failed: Expected {expected_result_2}, got {extract_entities(test_text_2)}"

    # Test case 3: Text with miscellaneous entities which should not be included in the results
    print("Testing case [3/3] started.")
    test_text_3 = "Mount Everest is located in the Himalayas."
    expected_result_3 = []  # No persons or organizations expected
    assert extract_entities(test_text_3) == expected_result_3, f"Test case [3/3] failed: Expected {expected_result_3}, got {extract_entities(test_text_3)}"
    print("Testing finished.")

# Run the test function
test_extract_entities()