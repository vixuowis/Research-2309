# function_import --------------------

import torch
from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features(input_text: str):
    """
    Extract features from text or code using the pre-trained CodeBERT model.

    Args:
        input_text (str): The input text or code from which to extract features.

    Returns:
        torch.Tensor: The extracted features (embeddings) from the input text or code.
    """
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    tokens = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    model = AutoModel.from_pretrained("microsoft/codebert-base")
        
    # get the mean of the first 12 layers of the encoder outputs
    with torch.no_grad():
        out = model(**tokens, output_hidden_states=True)["hidden_states"][0:12]
    
    feats = [x.mean(dim=-2).squeeze() for x in out]
        
    return torch.stack(feats)

# test_function_code --------------------

def test_extract_features():
    """
    Test the extract_features function.
    """
    input_text = 'def hello_world():\n    print("Hello, world!")'
    embeddings = extract_features(input_text)
    assert embeddings is not None
    assert embeddings.size(0) > 0
    return 'All Tests Passed'


# call_test_function_code --------------------

if __name__ == '__main__':
    print(test_extract_features())