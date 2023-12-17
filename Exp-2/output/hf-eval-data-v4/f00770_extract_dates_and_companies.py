# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_dates_and_companies(text):
    """
    Extract dates and company names from the given text using Named Entity Recognition.

    Parameters:
        text (str): The text containing potential dates and company names.

    Returns:
        list: Extracted dates and company names.
    """
    tokenizer = AutoTokenizer.from_pretrained('Jean-Baptiste/camembert-ner')
    model = AutoModelForTokenClassification.from_pretrained('Jean-Baptiste/camembert-ner')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')
    results = nlp(text)

    entities = []
    for entity in results:
        if entity['entity_group'] in ['DATE', 'ORG']:
            entities.append(entity['word'])
    return entities

# test_function_code --------------------

def test_extract_dates_and_companies():
    print("Testing started.")
    # Test Case 1
    text = "Apple was founded on April 1, 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne."
    expected = ['April 1, 1976', 'Apple']
    results = extract_dates_and_companies(text)
    assert set(results) == set(expected), f"Test case failed: {results} != {expected}"

    # Test Case 2
    text = "Microsoft, established on April 4, 1975, is headquartered in Redmond."
    expected = ['April 4, 1975', 'Microsoft']
    results = extract_dates_and_companies(text)
    assert set(results) == set(expected), f"Test case failed: {results} != {expected}"

    print("Testing finished.")