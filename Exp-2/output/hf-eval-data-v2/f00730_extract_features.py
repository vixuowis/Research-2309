# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def extract_features(source_code_text):
    """
    This function is used to extract features from a given source code text using the 'microsoft/unixcoder-base' model.
    
    Args:
        source_code_text (str): The source code text from which to extract features.
    
    Returns:
        feature_matrix (torch.Tensor): The feature matrix derived as a matrix of embeddings using the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained('microsoft/unixcoder-base')
    model = AutoModel.from_pretrained('microsoft/unixcoder-base')
    inputs = tokenizer(source_code_text, return_tensors='pt')
    outputs = model(**inputs)
    feature_matrix = outputs.last_hidden_state
    return feature_matrix

# test_function_code --------------------

def test_extract_features():
    """
    This function is used to test the 'extract_features' function.
    It uses a sample source code text and checks if the output is a torch.Tensor.
    """
    source_code_text = '/* Your source code here */'
    feature_matrix = extract_features(source_code_text)
    assert isinstance(feature_matrix, torch.Tensor), 'The output should be a torch.Tensor.'

# call_test_function_code --------------------

test_extract_features()