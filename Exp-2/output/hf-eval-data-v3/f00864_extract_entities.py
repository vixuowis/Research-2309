# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_entities(text):
    """
    Extract the names of people, organizations, and locations mentioned in the given text.

    Args:
        text (str): The input text from which to extract entities.

    Returns:
        list: A list of dictionaries, each containing information about an entity found in the text.
    """
    tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')
    model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')
    ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)
    ner_results = ner_pipeline(text)
    return ner_results

# test_function_code --------------------

def test_extract_entities():
    """
    Test the extract_entities function.
    """
    test_text1 = 'Hello, my name is John Doe, and I work at Microsoft. Tomorrow, I'll be going to a conference in San Francisco.'
    test_text2 = 'Apple Inc. is planning to open a new store in New York.'
    test_text3 = 'The Eiffel Tower is a famous landmark in Paris, France.'

    assert len(extract_entities(test_text1)) > 0
    assert len(extract_entities(test_text2)) > 0
    assert len(extract_entities(test_text3)) > 0

    print('All Tests Passed')

# call_test_function_code --------------------

test_extract_entities()