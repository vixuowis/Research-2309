# function_import --------------------

from transformers import AutoModel, AutoTokenizer
import torch

# function_code --------------------

def extract_features_biobert(text: str) -> torch.Tensor:
    """
    Extract features from text using BioBERT model.

    Args:
        text (str): Input text from which features need to be extracted.

    Returns:
        torch.Tensor: Extracted features in the form of a tensor.
    """
    model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state

# test_function_code --------------------

def test_extract_features_biobert():
    """
    Test the function extract_features_biobert.
    """
    text = 'The patient has lung cancer.'
    features = extract_features_biobert(text)
    assert isinstance(features, torch.Tensor), 'The output should be a torch.Tensor.'
    assert features.shape[1] == len(text.split()), 'The number of tokens in the output should match the number of words in the input.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_features_biobert()