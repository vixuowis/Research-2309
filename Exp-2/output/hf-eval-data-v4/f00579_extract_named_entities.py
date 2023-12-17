# requirements_file --------------------

!pip install -U transformers>=4.0.0

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_named_entities(article_text):
    """
    Extract named entities from the given article text using BERT model.
    
    Args:
        article_text (str): A news article or any text from which to extract named entities.
        
    Returns:
        list: A list of named entities extracted from the text.
    """
    ner_model = pipeline('ner', model='dslim/bert-base-NER-uncased')
    entities = ner_model(article_text)
    return entities

# test_function_code --------------------

def test_extract_named_entities():
    print("Testing extract_named_entities function.")
    test_article = "Jack and Jill went to Capitol Hill."

    # Expected entities include Jack, Jill, and Capitol Hill among other possible entities.
    expected_entities = {'Jack', 'Jill', 'Capitol Hill'}
    extracted_entities = extract_named_entities(test_article)

    # Verify if the expected entities are in the extracted entities
    for entity_dict in extracted_entities:
        assert entity_dict['entity'] in expected_entities, f"Expected entity not found: {entity_dict['entity']}"

    print("All tests passed!")

# Run the tests
test_extract_named_entities()