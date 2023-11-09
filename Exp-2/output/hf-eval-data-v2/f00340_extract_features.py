# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features(entity_name):
    """
    Extract features from biomedical entity names using the pre-trained model 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'.

    Args:
        entity_name (str): The name of the biomedical entity.

    Returns:
        Tensor: The [CLS] embedding of the last layer.
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
    entity_name = 'covid infection'
    cls_embedding = extract_features(entity_name)

    assert cls_embedding is not None
    assert cls_embedding.size()[0] == 1

# call_test_function_code --------------------

test_extract_features()