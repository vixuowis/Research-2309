# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features_from_code(text_or_code:str) -> torch.Tensor:
    """
    Extracts features from the given text or code snippet using the CodeBERT model.

    Args:
        text_or_code (str): The textual input or piece of code for feature extraction.

    Returns:
        torch.Tensor: A tensor containing the extracted feature representations (embeddings).
    """
    model = AutoModel.from_pretrained('microsoft/codebert-base')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    inputs = tokenizer(text_or_code, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    return embeddings

# test_function_code --------------------

def test_extract_features_from_code():
    print("Testing started.")

    # example text for feature extraction
    example_code = 'def hello_world(): print("Hello, World!")'

    # testing case 1: feature extraction from the example code
    print("Testing case [1/1] started.")
    features = extract_features_from_code(example_code)
    assert features is not None and features.shape[0] == 1, f"Test case [1/1] failed: Expected non-None features with batch size 1, got {features}" 
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_features_from_code()