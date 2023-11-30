# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def detect_named_entities(text):
    """
    Detect named entities in a given text using a multilingual named entity recognition model.

    Args:
        text (str): The text in which to detect named entities.

    Returns:
        list: A list of dictionaries, each containing information about a detected named entity.
    """ 

    model = AutoModelForTokenClassification.from_pretrained(
        "dbmdz/bert-large-cased-finetuned-conll03en-ner"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "dbmdz/bert-large-cased-finetuned-conll03en-ner", 
        use_fast=True,
    )

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    return nlp(text)

# test_function_code --------------------

def test_detect_named_entities():
    """
    Test the detect_named_entities function.
    """
    test_text_1 = 'Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute.'
    test_text_2 = 'Apple Inc. is planning to open a new store in San Francisco.'
    test_text_3 = 'Angela Merkel met with Emmanuel Macron in Berlin.'
    assert isinstance(detect_named_entities(test_text_1), list)
    assert isinstance(detect_named_entities(test_text_2), list)
    assert isinstance(detect_named_entities(test_text_3), list)
    print('All Tests Passed')


# call_test_function_code --------------------

test_detect_named_entities()