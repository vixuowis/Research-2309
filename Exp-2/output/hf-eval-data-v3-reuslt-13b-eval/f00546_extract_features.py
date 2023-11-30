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

    # Load SapBERT model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    
    # Create input tensor
    inputs = tokenizer.batch_encode_plus([entity_names], return_tensors="pt", padding=True, truncation=True)
    for key in inputs:
        inputs[key] = inputs[key].to(device)
        
    with torch.no_grad():
        # Extract [CLS] embedding from the model output
        out = model(**inputs)["last_hidden_state"][:, 0, :]
    
    return out

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