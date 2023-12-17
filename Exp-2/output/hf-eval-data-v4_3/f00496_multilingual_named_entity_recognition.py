# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# function_code --------------------

def multilingual_named_entity_recognition(text, model_name='Babelscape/wikineural-multilingual-ner'):
    """
    Perform multilingual named entity recognition on text using a pre-trained model.

    Args:
        text (str): The text to analyze for named entity recognition.
        model_name (str): The name of the pre-trained NER model.

    Returns:
        list: A list of named entity recognition results.

    Raises:
        ValueError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)

    return nlp(text)

# test_function_code --------------------

def test_multilingual_named_entity_recognition():
    print("Testing started.")
    
    # Define test cases
    test_cases = [
        {"text": "My name is Alicia and I live in Madrid.", "expected_number_of_entities": 2},
        {"text": "Ich heiße Wolfgang und ich wohne in Berlin.", "expected_number_of_entities": 2},
        {"text": "", "expected_number_of_entities": 0}  # Empty string test case
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"Testing case [{i}/{len(test_cases)}] started.")
        results = multilingual_named_entity_recognition(test_case["text"])
        assert len(results) == test_case["expected_number_of_entities"], f"Test case [{i}/{len(test_cases)}] failed: Expected {test_case['expected_number_of_entities']} entities, got {len(results)} in results {results}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_multilingual_named_entity_recognition()