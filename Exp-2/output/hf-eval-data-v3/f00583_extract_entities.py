# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_entities(text):
    """
    Extracts the names of companies and people mentioned in the text using Named Entity Recognition.

    Args:
        text (str): The text from which to extract entities.

    Returns:
        list: A list of entities (people and companies) extracted from the text.

    Raises:
        OSError: If there is an error in loading the NER model.
    """
    try:
        ner_model = pipeline('ner', model='Jean-Baptiste/roberta-large-ner-english')
        ner_results = ner_model(text)
        entities = [result['word'] for result in ner_results if result['entity'] in ['PER', 'ORG']]
        return entities
    except OSError as e:
        print(f'Error in loading the NER model: {e}')

# test_function_code --------------------

def test_extract_entities():
    """
    Tests the function extract_entities.
    """
    # Test case 1: Text with names of people and companies
    text1 = 'Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne to develop and sell Wozniak\'s Apple I personal computer.'
    assert set(extract_entities(text1)) == set(['Apple', 'Steve', 'Jobs', 'Steve', 'Wozniak', 'Ronald', 'Wayne']), 'Test case 1 failed'

    # Test case 2: Text with no names of people or companies
    text2 = 'The quick brown fox jumps over the lazy dog.'
    assert extract_entities(text2) == [], 'Test case 2 failed'

    # Test case 3: Text with names of people only
    text3 = 'Barack Obama was the 44th President of the United States.'
    assert set(extract_entities(text3)) == set(['Barack', 'Obama']), 'Test case 3 failed'

    print('All tests passed')

# call_test_function_code --------------------

test_extract_entities()