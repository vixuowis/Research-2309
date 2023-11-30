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
    # Load the tokenizer and model.
    sapbert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    sapbert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', from_tf=True, output_hidden_states=False)

    # Prepare the input format for the SapBERT model.
    encoded_input = sapbert_tokenizer(entity_names, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    
    # Extract the [CLS] embedding of the biomedical entity names from the model output.
    with torch.no_grad():
        sapbert_model_output = sapbert_model(**encoded_input)
        embedded_entity_names = sapbert_model_output[1][:, 0, :]
    
    return embedded_entity_names

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