# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_features_from_entities(entity_names):
    """
    Extracts features from a list of biomedical entity names using SapBERT.

    Parameters:
        entity_names (list): A list of strings, where each string is a biomedical entity name.

    Returns:
        torch.Tensor: The [CLS] embedding representing the aggregated features of the input
    """
    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

    inputs = tokenizer(entity_names, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# test_function_code --------------------

def test_extract_features_from_entities():
    print("Testing started.")
    # Test with a sample list of biomedical entity names
    entity_names = ['covid infection', 'diabetes mellitus', 'hypertension']

    # Expected shape of the [CLS] embedding
    expected_shape = (len(entity_names), model.config.hidden_size)

    print("Test case [1/1] started.")
    cls_embedding = extract_features_from_entities(entity_names)
    assert cls_embedding.shape == expected_shape, f"Test case [1/1] failed: Expected shape {expected_shape}, got {cls_embedding.shape}"
    print("Testing finished.")

test_extract_features_from_entities()