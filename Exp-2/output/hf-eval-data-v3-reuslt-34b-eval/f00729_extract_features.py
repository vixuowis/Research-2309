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
    model = AutoModel.from_pretrained(
        "microsoft/codebert-base",
        output_attentions=False,
        output_hidden_states=False)

    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model(**inputs)

    # Extract features from the last layer of the pre-trained transformer model.
    embeddings = outputs.last_hidden_state[0]  # shape: [12 x hidden_dim]
    feature_vec = torch.mean(embeddings, dim=0)

    return feature_vec


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