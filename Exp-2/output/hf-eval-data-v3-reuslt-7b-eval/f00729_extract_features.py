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
    tokenizer = AutoTokenizer.from_pretrained('microsoft/CodeBERT-base')
    model = AutoModel.from_pretrained("microsoft/CodeBERT-base")
    
    input = tokenizer([input_text], return_tensors='pt').to(device)
    outputs = model(**input, output_hidden_states=True)

    last_hidden_state = outputs.last_hidden_state # [1 x Length x 1024]

    features = last_hidden_state[:,0,:]
    return features

# function_main --------------------

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