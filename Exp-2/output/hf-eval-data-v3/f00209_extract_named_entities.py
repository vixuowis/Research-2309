# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_named_entities(text):
    """
    Extract named entities from the given text using a pre-trained model.

    Args:
        text (str): The text from which to extract named entities.

    Returns:
        list: A list of dictionaries. Each dictionary represents a named entity and contains the entity, its start and end indices in the text, and its type.
    """
    tokenizer = AutoTokenizer.from_pretrained('Babelscape/wikineural-multilingual-ner')
    model = AutoModelForTokenClassification.from_pretrained('Babelscape/wikineural-multilingual-ner')
    ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)
    return ner_pipeline(text)

# test_function_code --------------------

def test_extract_named_entities():
    assert len(extract_named_entities('My name is Wolfgang and I live in Berlin. Mi nombre es José y vivo en Madrid.')) > 0
    assert len(extract_named_entities('')) == 0
    assert len(extract_named_entities('This is a test sentence with no named entities.')) == 0
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_named_entities()