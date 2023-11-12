# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_features(entity_names):
    """
    Extract features from a set of entity names using the SapBERT model.

    Args:
        entity_names (str): A string of biomedical entity names.

    Returns:
        torch.Tensor: The [CLS] embedding from the model output, which represents the aggregated features for the input biomedical entity names.
    """
    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

    inputs = tokenizer(entity_names, return_tensors='pt')
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding

# test_function_code --------------------

def test_extract_features():
    """
    Test the extract_features function.
    """
    entity_names = 'covid infection'
    cls_embedding = extract_features(entity_names)

    assert cls_embedding.shape[0] == 1
    assert cls_embedding.shape[1] == 768

    entity_names = 'cancer cell'
    cls_embedding = extract_features(entity_names)

    assert cls_embedding.shape[0] == 1
    assert cls_embedding.shape[1] == 768

    entity_names = 'heart disease'
    cls_embedding = extract_features(entity_names)

    assert cls_embedding.shape[0] == 1
    assert cls_embedding.shape[1] == 768

    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_features()