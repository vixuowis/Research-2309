# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_features_from_code(source_code_text):
    """Extracts a feature matrix from the given source code text using the UniXcoder model.

    Args:
        source_code_text (str): The source code text to be tokenized and analyzed.

    Returns:
        torch.Tensor: A tensor representing the feature matrix of the source code text.

    Raises:
        ValueError: If the 'source_code_text' is not a string or is empty.
    """
    if not isinstance(source_code_text, str) or not source_code_text:
        raise ValueError("The source code text must be a non-empty string.")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/unixcoder-base')
    model = AutoModel.from_pretrained('microsoft/unixcoder-base')
    inputs = tokenizer(source_code_text, return_tensors='pt')
    outputs = model(**inputs)
    feature_matrix = outputs.last_hidden_state
    return feature_matrix


# test_function_code --------------------

def test_extract_features_from_code():
    print("Testing started.")
    sample_code = 'def example_function(): pass'  # Example source code

    # Test case 1: Valid source code
    print("Testing case [1/1] started.")
    feature_matrix = extract_features_from_code(sample_code)
    assert feature_matrix is not None and feature_matrix.nelement() > 0, f"Test case [1/1] failed: Feature matrix is None or empty"
    print("Testing finished.")


# call_test_function_line --------------------

test_extract_features_from_code()