# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def multilingual_ner(text):
    """
    This function uses a pre-trained Named Entity Recognition (NER) model to extract entities from a given text.
    The model supports 9 languages (de, en, es, fr, it, nl, pl, pt, ru).

    Args:
        text (str): The text from which to extract entities.

    Returns:
        List[Dict]: A list of dictionaries. Each dictionary represents an entity and contains the entity text, start index, end index, and entity type.
    """
    tokenizer = AutoTokenizer.from_pretrained('Babelscape/wikineural-multilingual-ner')
    model = AutoModelForTokenClassification.from_pretrained('Babelscape/wikineural-multilingual-ner')
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    return nlp(text)

# test_function_code --------------------

def test_multilingual_ner():
    """
    This function tests the multilingual_ner function by passing a sample text and checking the output.
    """
    example = "My name is Wolfgang and I live in Berlin."
    ner_results = multilingual_ner(example)
    assert isinstance(ner_results, list), "The output should be a list."
    assert 'entity' in ner_results[0], "Each item in the list should be a dictionary with an 'entity' key."
    assert 'index' in ner_results[0], "Each item in the list should be a dictionary with an 'index' key."
    assert 'start' in ner_results[0], "Each item in the list should be a dictionary with a 'start' key."
    assert 'end' in ner_results[0], "Each item in the list should be a dictionary with an 'end' key."

# call_test_function_code --------------------

test_multilingual_ner()