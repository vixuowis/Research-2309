# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def detect_named_entities(text):
    """
    Detect named entities in a given text using a multilingual named entity recognition model.

    Args:
        text (str): The text in which to detect named entities.

    Returns:
        List[Dict]: A list of dictionaries, each containing information about a detected named entity.
    """
    tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    return nlp(text)

# test_function_code --------------------

def test_detect_named_entities():
    """
    Test the detect_named_entities function.
    """
    example = "Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute."
    ner_results = detect_named_entities(example)
    assert isinstance(ner_results, list)
    assert 'entity' in ner_results[0]
    assert 'score' in ner_results[0]
    assert 'index' in ner_results[0]
    assert 'start' in ner_results[0]
    assert 'end' in ner_results[0]

# call_test_function_code --------------------

test_detect_named_entities()