# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification

# function_code --------------------

def biomedical_entity_recognition(text):
    """
    Recognizes biomedical entities from a given text using a pretrained model.

    Args:
        text (str): The text from which to recognize biomedical entities.

    Returns:
        dict: A dictionary containing the recognized entities and their corresponding scores.
    """
    tokenizer = AutoTokenizer.from_pretrained('d4data/biomedical-ner-all')
    model = AutoModelForTokenClassification.from_pretrained('d4data/biomedical-ner-all')
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs

# test_function_code --------------------

def test_biomedical_entity_recognition():
    """
    Tests the biomedical_entity_recognition function.
    """
    test_text = 'The patient reported no recurrence of palpitations at follow-up 6 months after the ablation.'
    result = biomedical_entity_recognition(test_text)
    assert 'entities' in result, 'Result should contain recognized entities.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_biomedical_entity_recognition()