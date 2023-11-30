# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_features(source_code_text):
    """
    Extracts features from the given source code text using the 'microsoft/unixcoder-base' model.

    Args:
        source_code_text (str): The source code text to extract features from.

    Returns:
        torch.Tensor: The feature matrix derived as a matrix of embeddings.
    """
    
    # Get model tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = AutoModel.from_pretrained("microsoft/unixcoder-base", return_dict=False)

    # Tokenize the source code
    tokens = tokenizer(source_code_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    
    with torch.no_grad():
        # Compute model output and embeddings
        outputs = model(**tokens)
        embedding = outputs[1]
        
        # Get feature matrix (embedding matrix of the last hidden-state, averaged across token sequence)
        feature_matrix = torch.mean(embedding, dim=1).squeeze()
    
    return feature_matrix

# test_function_code --------------------

def test_extract_features():
    """
    Tests the 'extract_features' function.
    """
    source_code_text = '/* Your source code here */'
    feature_matrix = extract_features(source_code_text)
    assert isinstance(feature_matrix, torch.Tensor), 'The output should be a torch.Tensor.'
    print('All Tests Passed')


# call_test_function_code --------------------

if __name__ == '__main__':
    test_extract_features()