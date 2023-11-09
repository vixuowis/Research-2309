import torch

# Function to test the feature extraction function
def test_extract_features_biobert():
    """
    This function tests the extract_features_biobert function.
    It uses a sample biomedical text and checks if the output is a tensor.
    """
    # Sample biomedical text
    text = 'The patient was diagnosed with lung cancer.'
    
    # Extract features
    features = extract_features_biobert(text)
    
    # Check if the output is a tensor
    assert isinstance(features, torch.Tensor), 'The output should be a tensor.'
    
    # Check if the output tensor is not empty
    assert features.nelement() > 0, 'The output tensor should not be empty.'

test_extract_features_biobert()