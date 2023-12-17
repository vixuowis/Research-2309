# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_meeting_locations(text):
    """
    Extracts meeting locations from a given multilingual text.

    Parameters:
    text (str): The multilingual text from which to extract meeting locations.

    Returns:
    list: A list of extracted locations.
    """
    tokenizer = AutoTokenizer.from_pretrained('Babelscape/wikineural-multilingual-ner')
    model = AutoModelForTokenClassification.from_pretrained('Babelscape/wikineural-multilingual-ner')

    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    ner_results = nlp(text)

    # Filter out only location entities
    locations = [entity['word'] for entity in ner_results if entity['entity_group'] == 'LOC']
    return locations

# test_function_code --------------------

def test_extract_meeting_locations():
    print("Testing extract_meeting_locations function.")

    # Test case 1: Single language, single location
    text1 = "The next conference will be held in Berlin."
    assert 'Berlin' in extract_meeting_locations(text1), "Test case 1 failed: Berlin was not extracted as a location."

    # Test case 2: Multilingual, multiple locations
    text2 = "We have two seminars: one in Paris and another in Madrid."
    result = extract_meeting_locations(text2)
    assert 'Paris' in result and 'Madrid' in result, "Test case 2 failed: Locations Paris and Madrid were not extracted."

    # Test case 3: No locations provided
    text3 = "Our meeting will be conducted online."
    assert not extract_meeting_locations(text3), "Test case 3 failed: No locations should be extracted."

    print("All test cases passed!")