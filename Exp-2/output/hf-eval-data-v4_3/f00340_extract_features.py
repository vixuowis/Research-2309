# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features(text: str) -> torch.Tensor:
    """
    Extract features from a biomedical entity name using SapBERT.

    Args:
        text: The string of the biomedical entity name to be processed.

    Returns:
        A PyTorch tensor of the [CLS] embedding of the last layer from the model.

    Raises:
        ValueError: If 'text' is not provided.
    """
    if not text:
        raise ValueError("'text' must be provided.")
    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# test_function_code --------------------

def test_extract_features():
    print("Testing started.")

    # Test case 1: Provide a valid biomedical entity name
    print("Testing case [1/3] started.")
    entity_name = 'covid infection'
    assert extract_features(entity_name) is not None, "Test case [1/3] failed: The function should return a tensor."

    # Test case 2: Providing an empty string should raise ValueError
    print("Testing case [2/3] started.")
    try:
        extract_features('')
        assert False, "Test case [2/3] failed: The function should raise ValueError when text is empty."
    except ValueError:
        assert True

    # Test case 3: Providing None should raise ValueError
    print("Testing case [3/3] started.")
    try:
        extract_features(None)
        assert False, "Test case [3/3] failed: The function should raise ValueError when text is None."
    except ValueError:
        assert True
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_features()