# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_named_entities(text):
    """
    Extract names, organizations, and locations from the given text using
    a Named Entity Recognition (NER) model.

    Parameters:
        text (str): The text from which to extract entities.

    Returns:
        list: A list of extracted entities with their types and positions in the text.
    """
    tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')
    model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')
    ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)
    ner_results = ner_pipeline(text)
    return ner_results

# test_function_code --------------------

def test_extract_named_entities():
    print("Testing extract_named_entities function.")
    input_text = "Hello, my name is John Doe, and I work at Microsoft. Tomorrow, I'll be going to a conference in San Francisco."
    expected_entities = [
        {'word': 'John Doe', 'entity': 'B-PER'},
        {'word': 'Microsoft', 'entity': 'B-ORG'},
        {'word': 'San Francisco', 'entity': 'B-LOC'}
    ]
    entities = extract_named_entities(input_text)
    
    # Test case: Check if the expected entities are in the results.
    assert all(item in entities for item in expected_entities), \
        "The extracted entities do not match the expected entities."
    print("Test passed.")
