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
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT-small-v1")
    model = AutoModel.from_pretrained("microsoft/CodeBERT-small-v1")
    model.eval()

    # Split the code into a list of lines
    source_code_text = source_code_text.split("\n")[2::] # Removes header
    
    # Tokenize the code and add it to the input matrix
    inputs_dict = tokenizer(source_code_text, padding=True, truncation=True, return_tensors="pt")

    # Extract features from the model
    with torch.no_grad():
        outputs = model(**inputs_dict)
    
    # Return the average of all code tokens as the embedding for that line
    return torch.mean(outputs[0], 1).cpu()


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