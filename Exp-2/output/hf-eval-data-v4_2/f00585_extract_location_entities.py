# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_location_entities(text):
    """
    Extract location entities from a given multilingual text using a pre-trained NER model.

    Args:
        text (str): The text from which to extract location entities.

    Returns:
        list: A list containing the extracted location entities.

    Raises:
        ValueError: If the input text is empty.
    """
    if not text:
        raise ValueError('Input text cannot be empty.')

    tokenizer = AutoTokenizer.from_pretrained('Babelscape/wikineural-multilingual-ner')
    model = AutoModelForTokenClassification.from_pretrained('Babelscape/wikineural-multilingual-ner')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    ner_results = nlp(text)

    # Extract locations from the results
    locations = [entity['word'] for entity in ner_results if entity['entity_group'] == 'LOC']
    return locations

# test_function_code --------------------

def test_extract_location_entities():
    print('Testing started.')
    # Use an example multilingual text with known locations
    example_text = 'My name is Wolfgang and I live in Berlin. Mi nombre es Carlos y vivo en Madrid.'

    # Test case 1: Detect locations in English and Spanish
    print('Testing case [1/2] started.')
    expected_locations = ['Berlin', 'Madrid']
    actual_locations = extract_location_entities(example_text)
    assert set(actual_locations) == set(expected_locations), f'Test case [1/2] failed: expected {expected_locations}, got {actual_locations}'

    # Test case 2: Raise ValueError when input text is empty
    print('Testing case [2/2] started.')
    try:
        extract_location_entities('')
        assert False, 'Test case [2/2] failed: ValueError not raised for empty text.'
    except ValueError as e:
        assert str(e) == 'Input text cannot be empty.', f'Test case [2/2] failed: {e}'
    print('Testing finished.')

# call_test_function_line --------------------

test_extract_location_entities()