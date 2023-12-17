# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, pipeline

# function_code --------------------

def extract_dates_and_companies(text):
    """
    Extracts dates and company names from a given text using a pre-trained NER model.

    Args:
        text (str): The text from which to extract dates and company names.

    Returns:
        List[dict]: A list of dictionaries with the extracted entities and their types.

    Raises:
        ValueError: If the text is empty.
    """
    if not text:
        raise ValueError('The input text is empty')
    tokenizer = AutoTokenizer.from_pretrained('Jean-Baptiste/camembert-ner')
    model = AutoModelForTokenClassification.from_pretrained('Jean-Baptiste/camembert-ner')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')
    results = nlp(text)

    # Filter out only date and company entities
    entities = [entity for entity in results if entity['entity_group'] in ['DATE', 'ORG']]
    return entities

# test_function_code --------------------

def test_extract_dates_and_companies():
    print("Testing started.")
    # Define a test case with known entities
    test_text = "Apple was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne."
    expected_entities = [
        {'entity_group': 'DATE', 'word': "April 1, 1976"},
        {'entity_group': 'ORG', 'word': 'Apple'}
    ]

    # Test case 1: Text with known entities
    print("Testing case [1/1] started.")
    extracted_entities = extract_dates_and_companies(test_text)
    assert extracted_entities == expected_entities, f"Test case [1/1] failed: expected {expected_entities}, got {extracted_entities}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_dates_and_companies()