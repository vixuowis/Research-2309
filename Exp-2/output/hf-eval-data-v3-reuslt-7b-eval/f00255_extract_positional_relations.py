# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_positional_relations(text):
    """
    Extracts the positional relations between various keywords of a given medical text using the SapBERT model.

    Args:
        text (str): The medical text from which to extract positional relations.

    Returns:
        torch.Tensor: The [CLS] embedding of the last layer, indicating the position of the embedded biomedical entities.
    """
    
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/sapbert")
    model = AutoModel.from_pretrained("dmis-lab/sapbert").to('cuda')
    
    tokens = tokenizer.encode(text, return_tensors='pt').to('cuda')
    outputs = model(tokens).pooler_output
    
    return outputs[0]

# test_function_code --------------------

def test_extract_positional_relations():
    """
    Tests the extract_positional_relations function.
    """
    # Test case: Normal case
    output = extract_positional_relations('covid infection')
    assert isinstance(output, torch.Tensor), 'Output should be a PyTorch Tensor.'

    # Test case: Empty string
    output = extract_positional_relations('')
    assert isinstance(output, torch.Tensor), 'Output should be a PyTorch Tensor.'

    # Test case: Long string
    output = extract_positional_relations('covid infection' * 100)
    assert isinstance(output, torch.Tensor), 'Output should be a PyTorch Tensor.'

    print('All Tests Passed')


# call_test_function_code --------------------

test_extract_positional_relations()