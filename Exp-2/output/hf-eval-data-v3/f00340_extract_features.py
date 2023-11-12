# function_import --------------------

import torch
from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features(entity_name):
    """
    Extract features from biomedical entity names using the pre-trained model 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'.

    Args:
        entity_name (str): The biomedical entity name to extract features from.

    Returns:
        torch.Tensor: The [CLS] embedding of the last layer of the model output.
    """
    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

    inputs = tokenizer(entity_name, return_tensors='pt')
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding

# test_function_code --------------------

def test_extract_features():
    """
    Test the function extract_features.
    """
    entity_names = ['covid infection', 'heart disease', 'diabetes']
    for entity_name in entity_names:
        cls_embedding = extract_features(entity_name)
        assert cls_embedding.shape[0] == 1, 'The output shape is not correct.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_features()