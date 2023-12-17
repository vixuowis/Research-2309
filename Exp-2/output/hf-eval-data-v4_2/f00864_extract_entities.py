# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def extract_entities(text):
    """
    Extract names of people, organizations, and locations from the given text using NER.

    Args:
        text (str): The input text from which to extract entities.

    Returns:
        List[dict]: A list of dictionaries where each dict represents an entity extracted from the text.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('The input text must be a string.')

    tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')
    model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')
    ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)

    ner_results = ner_pipeline(text)
    return ner_results

# test_function_code --------------------

def test_extract_entities():
    print("Testing started.")
    test_cases = [
        ("John Doe works at Microsoft.", 2),
        ("The Eiffel Tower is in Paris.", 2),
        ("", 0)  # Empty string test case
    ]

    for i, (text, expected_count) in enumerate(test_cases, start=1):
        print(f"Testing case [{i}/{len(test_cases)}] started.")
        entities = extract_entities(text)
        assert len(entities) == expected_count, f"Test case [{i}/{len(test_cases)}] failed: expected {expected_count}, got {len(entities)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_entities()