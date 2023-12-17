# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def detect_named_entities(text):
    """
    Detects named entities in a given text using a multilingual NER model.

    Args:
        text (str): The input text in which to detect named entities.

    Returns:
        List[Dict]: A list of dictionaries containing the detected entities.

    Raises:
        ValueError: If the text is empty or not a string.

    """
    # Validate input
    if not text or not isinstance(text, str):
        raise ValueError('Input text must be a non-empty string.')

    # Load the pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
    model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')

    # Create a pipeline for named entity recognition
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)

    # Perform NER on the input text
    return nlp(text)

# test_function_code --------------------

def test_detect_named_entities():
    print("Testing started.")
    # Test case: Detect entities in English
    print("Testing case [1/2] started.")
    english_text = "Nader Jokhadar has given Syria the lead with a well-struck header in the seventh minute."
    english_results = detect_named_entities(english_text)
    assert type(english_results) is list and len(english_results) > 0, f"Test case [1/2] failed: {english_results}"

    # Test case: Detect entities in Spanish
    print("Testing case [2/2] started.")
    spanish_text = "La ciudad de Barcelona es la capital de Cataluña."
    spanish_results = detect_named_entities(spanish_text)
    assert type(spanish_results) is list and len(spanish_results) > 0, f"Test case [2/2] failed: {spanish_results}"
    print("Testing finished.")

# call_test_function_line --------------------

test_detect_named_entities()