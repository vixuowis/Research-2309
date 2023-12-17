# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModel
import torch

# function_code --------------------

def extract_features_and_labels(code_segments):
    """
    Extracts features and labels from code segments using a pre-trained model.

    Args:
        code_segments (list of str): A list of code segments to process.

    Returns:
        dict: A dictionary with two keys 'features' and 'labels', containing the
             extracted features as Tensors and the associated labels.

    Raises:
        ValueError: If code_segments is not a list or is empty.
    """
    # Validate input
    if not isinstance(code_segments, list) or not code_segments:
        raise ValueError('code_segments must be a non-empty list')

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained('microsoft/unixcoder-base')
    model = AutoModel.from_pretrained('microsoft/unixcoder-base')

    # Tokenize and encode the code segments
    inputs = tokenizer(code_segments, return_tensors='pt', padding=True, truncation=True)

    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.last_hidden_state

    # Extract labels (dummy labels for demonstration purposes)
    labels = torch.ones(len(code_segments))

    return {'features': features, 'labels': labels}

# test_function_code --------------------

def test_extract_features_and_labels():
    print("Testing started.")
    code_segments = [
        'def hello_world(): print("Hello, world!")',
        '# This is a comment',
        'import numpy as np'
    ]  # A sample list of code segments

    # Test case 1: Valid input
    print("Testing case [1/2] started.")
    result = extract_features_and_labels(code_segments)
    assert type(result['features']) == torch.Tensor and type(result['labels']) == torch.Tensor, f"Test case [1/2] failed: Expected tensors, but got {type(result['features'])} and {type(result['labels'])}"

    # Test case 2: Invalid input
    print("Testing case [2/2] started.")
    try:
        extract_features_and_labels([])  # Empty list should raise ValueError
        assert False, "Test case [2/2] failed: ValueError not raised for empty input list"
    except ValueError:
        pass  # Expected behavior
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_features_and_labels()