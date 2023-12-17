# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_named_entities(text, model='dslim/bert-base-NER-uncased'):
    """
    Predict the named entities in the given text using the specified model.

    :param text: The text to analyze for named entities.
    :param model: The pre-trained NER model (default is 'dslim/bert-base-NER-uncased').
    :return: A list of named entity predictions.
    """
    nlp = pipeline('ner', model=model)
    return nlp(text)

# test_function_code --------------------

def test_predict_named_entities():
    print("Testing the predict_named_entities function.")

    # Test with a known sentence
    sentence = "Hawking was born on January 8, 1942, in Oxford, England."
    expected_entities = [{'entity_group': 'PER', 'score': 0.999, 'word': 'Hawking', 'start': 0, 'end': 7}, {'entity_group': 'DATE', 'score': 0.999, 'word': 'January 8, 1942', 'start': 19, 'end': 34}, {'entity_group': 'LOC', 'score': 0.999, 'word': 'Oxford', 'start': 38, 'end': 44}, {'entity_group': 'LOC', 'score': 0.999, 'word': 'England', 'start': 46, 'end': 53}]
    entities = predict_named_entities(sentence)
    assert len(entities) == len(expected_entities), f"Test failed: Expected number of entities {len(expected_entities)}, but got {len(entities)}."
    print("Testing finished successfully.")