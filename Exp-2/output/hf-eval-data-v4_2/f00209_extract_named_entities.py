# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_named_entities(text):
    """
    Extract named entities from the given text using a multilingual NER model.

    Args:
        text (str): Text from which to extract entities.

    Returns:
        list: A list of dictionaries with extracted entities and their types.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')

    tokenizer = AutoTokenizer.from_pretrained('Babelscape/wikineural-multilingual-ner')
    model = AutoModelForTokenClassification.from_pretrained('Babelscape/wikineural-multilingual-ner')
    ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)

    return ner_pipeline(text)

# test_function_code --------------------

def test_extract_named_entities():
    print("Testing started.")
    # Test case: A multilingual text
    print("Testing case [1/3] started.")
    test_text_1 = 'My name is Wolfgang and I live in Berlin. Mi nombre es José y vivo en Madrid.'
    expected_entities_1 = ['Wolfgang', 'Berlin', 'José', 'Madrid']
    result_1 = extract_named_entities(test_text_1)
    assert all(entity['word'] in result_1 for entity in expected_entities_1), f"Test case [1/3] failed: {result_1}"

    # Test case: An empty string
    print("Testing case [2/3] started.")
    test_text_2 = ''
    expected_entities_2 = []
    result_2 = extract_named_entities(test_text_2)
    assert result_2 == expected_entities_2, f"Test case [2/3] failed: {result_2}"

    # Test case: Non-string input
    print("Testing case [3/3] started.")
    test_text_3 = None
    try:
        extract_named_entities(test_text_3)
        assert False, "Test case [3/3] failed: No exception raised for non-string input."
    except ValueError as e:
        assert str(e) == 'Input text must be a string.', f"Test case [3/3] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_named_entities()