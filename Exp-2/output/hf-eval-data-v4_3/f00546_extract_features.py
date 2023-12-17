# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_features(entities):
    """
    Extracts the [CLS] embedding features from a list of biomedical entity names using the SapBERT model.

    Args:
        entities (list of str): A list of biomedical entity names.

    Returns:
        torch.Tensor: The [CLS] embeddings for the list of entities.

    Raises:
        ValueError: If entities are not provided as a list or if the list is empty.
    """
    if not isinstance(entities, list) or not entities:
        raise ValueError('The entities should be a non-empty list of strings.')

    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

    inputs = tokenizer(entities, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings

# test_function_code --------------------

def test_extract_features():
    print("Testing started.")

    # Test case 1: Testing with actual biomedical entities
    entities = ['covid infection', 'myocardial infarction', 'malignant neoplasm']
    print("Testing case [1/2] started.")
    embeddings = extract_features(entities)
    assert embeddings.shape[0] == len(entities), f"Test case [1/2] failed: Expected {len(entities)} embeddings but got {embeddings.shape[0]}"

    # Test case 2: Testing with an empty list, should raise ValueError
    print("Testing case [2/2] started.")
    try:
        extract_features([])
        assert False, "Test case [2/2] failed: ValueError was not raised for an empty list"
    except ValueError:
        assert True

    print("Testing finished.")

# call_test_function_line --------------------

test_extract_features()