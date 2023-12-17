# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForTokenClassification, AutoTokenizer

# function_code --------------------

def extract_entities(text, model_name='ismail-lucifer011/autotrain-job_all-903929564', use_auth_token=True):
    """
    Extract named entities such as names and locations from a given text using NLP token classification.

    Args:
        text (str): Text from which to extract named entities.
        model_name (str): The pre-trained model to use for token classification.
        use_auth_token (bool): Whether to use authentication token for loading the model.

    Returns:
        list: A list containing the extracted named entities.

    Raises:
        ValueError: If the provided text is empty.
    """
    if not text:
        raise ValueError('Input text cannot be empty.')

    # Load the model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_name, use_auth_token=use_auth_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)

    # Tokenize the input text and pass it to the model
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)

    # Extract the entities
    entities = tokenizer.convert_ids_to_tokens(outputs.argmax(dim=2).squeeze().tolist())
    names_and_locations = [token for token, label in zip(entities, outputs.argmax(dim=2).squeeze().tolist()) if label in {'location_label_id', 'name_label_id'}]

    return names_and_locations

# test_function_code --------------------

def test_extract_entities():
    print("Testing started.")
    # Test case 1: Valid text with entities
    print("Testing case [1/3] started.")
    sample_text = 'John is from New York and works at Google.'
    extracted_entities = extract_entities(sample_text)
    assert 'John' in extracted_entities and 'New York' in extracted_entities, f"Test case [1/3] failed: {extracted_entities}"

    # Test case 2: Text with no entities
    print("Testing case [2/3] started.")
    sample_text = 'Hello world!'
    extracted_entities = extract_entities(sample_text)
    assert not extracted_entities, f"Test case [2/3] failed: Entities found in text with no entities: {extracted_entities}"

    # Test case 3: Empty text
    print("Testing case [3/3] started.")
    sample_text = ''
    try:
        extract_entities(sample_text)
        assert False, 'Test case [3/3] failed: No error raised for empty text.'
    except ValueError as e:
        assert str(e) == 'Input text cannot be empty.', f'Test case [3/3] failed: Wrong error message for empty text: {e}'
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_entities()