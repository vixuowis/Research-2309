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
    
    # Tokenize the entity name list using SapBERT's tokenizer. This will also include a special "<eos>" token as the last element of each list.
    tokens = [model_tokenizer(name, add_special_tokens=True, truncation='do_not_truncate', max_length=maxlen) for name in entity_names]
    
    # Pad all tokens to be the same length using SapBERT's tokenizer. Note that this will also add a special "<pad>" token as padding.
    input_ids = model_tokenizer.pad(tokens, return_tensors='pt')['input_ids']
    
    # Use the pre-trained SapBERT to extract features from the entity names.
    with torch.no_grad():
        outputs = model(**input_ids)
    
    # Return the [CLS] feature vector (i.e., the last layer activation of the transformer's embedding layer).
    return outputs[0][:, 0, :].clone().detach()

# function_call --------------------

model_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# The maximum length of a biomedical entity name for SapBERT is 250 tokens as defined by the paper.
maxlen = 250


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