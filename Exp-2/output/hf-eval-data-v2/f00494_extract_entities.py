# function_import --------------------

from transformers import AutoModelForTokenClassification, AutoTokenizer

# function_code --------------------

def extract_entities(text):
    """
    Extracts names and locations from a given text using a pre-trained token classification model.

    Args:
        text (str): The text from which to extract entities.

    Returns:
        list: A list of entities (names and locations) extracted from the text.
    """
    model = AutoModelForTokenClassification.from_pretrained('ismail-lucifer011/autotrain-job_all-903929564', use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained('ismail-lucifer011/autotrain-job_all-903929564', use_auth_token=True)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    entities = tokenizer.convert_ids_to_tokens(outputs.argmax(dim=2).squeeze().tolist())
    names_and_locations = [token for token, label in zip(entities, outputs.argmax(dim=2).squeeze().tolist()) if label in {"location_label_id", "name_label_id"}]

    return names_and_locations

# test_function_code --------------------

def test_extract_entities():
    """
    Tests the extract_entities function.
    """
    test_text = "John and Mary went to New York."
    expected_output = ['John', 'Mary', 'New York']
    assert set(extract_entities(test_text)) == set(expected_output), "Test failed!"

    test_text = "The Eiffel Tower is in Paris."
    expected_output = ['Eiffel Tower', 'Paris']
    assert set(extract_entities(test_text)) == set(expected_output), "Test failed!"

# call_test_function_code --------------------

test_extract_entities()