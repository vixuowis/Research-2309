# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# function_code --------------------

def extract_named_entities(text):
    """
    Extract names of people, organizations, and locations from the input text using a multilingual NER model.

    Parameters:
    text (str): The text to extract entities from.

    Returns:
    list: A list of dictionaries with entity type and extracted entity.
    """
    tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    return nlp(text)

# test_function_code --------------------

def test_extract_named_entities():
    print("Testing started.")
    example = "Angela Merkel and Barack Obama met in Berlin to discuss international policies."

    # Test case 1
    print("Testing case [1/1] started.")
    result = extract_named_entities(example)
    expected_entities = ['Angela Merkel', 'Barack Obama', 'Berlin']
    assert all(entity['word'] in expected_entities for entity in result), f"Test case failed: Expected entities not found in: {result}"
    print("Testing finished.")

# Run the test function
test_extract_named_entities()