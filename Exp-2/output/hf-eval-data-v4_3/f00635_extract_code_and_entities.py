# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# function_code --------------------

def extract_code_and_entities(text):
    """Extract code syntax and named entities from a text using a pre-trained BERTOverflow model.

    Args:
        text (str): Text from which to extract code and named entities.

    Returns:
        dict: Dictionary containing extracted code tokens and named entities.

    Raises:
        ValueError: If text input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input text must be a string.')

    # Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('lanwuwei/BERTOverflow_stackoverflow_github')
    model = AutoModelForTokenClassification.from_pretrained('lanwuwei/BERTOverflow_stackoverflow_github')

    # Tokenize text
    inputs = tokenizer(text, return_tensors='pt')
    # Get model predictions
    outputs = model(**inputs)
    # Process predictions
    predictions = torch.argmax(outputs.logits, dim=-1)
    # Map predictions to token strings
    tokens = [tokenizer.convert_ids_to_tokens(input_id) for input_id in inputs['input_ids'][0]]

    # Extract entities and code tokens
    extracted_data = {
        'entities': [],
        'code_tokens': []
    }
    for token, prediction in zip(tokens, predictions[0].numpy()):
        if prediction != 0:  # Ignore 'O' (Outside any named entity)
            extracted_data['entities' if prediction % 2 == 1 else 'code_tokens'].append(token)
    return extracted_data

# test_function_code --------------------

def test_extract_code_and_entities():
    print("Testing started.")
    sample_text = "How to extract elements from a list in Python?"

    # Test case 1: input is a string
    print("Testing case [1/3] started.")
    result = extract_code_and_entities(sample_text)
    assert isinstance(result, dict), f"Test case [1/3] failed: result should be a dict, got {type(result)}"

    # Test case 2: input is not a string
    print("Testing case [2/3] started.")
    try:
        extract_code_and_entities(None)
        assert False, "Test case [2/3] failed: ValueError expected."
    except ValueError as e:
        assert str(e) == 'Input text must be a string.', f"Test case [2/3] failed: {str(e)}"

    # Test case 3: check entities and code tokens extraction
    print("Testing case [3/3] started.")
    assert 'entities' in result and 'code_tokens' in result, f"Test case [3/3] failed: keys 'entities' and 'code_tokens' expected in result."
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_code_and_entities()