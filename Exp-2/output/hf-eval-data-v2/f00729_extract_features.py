# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features(input_text):
    """
    Extract features from text and code using Transformer models.

    Args:
        input_text (str): The input text or code from which to extract features.

    Returns:
        torch.Tensor: The extracted features (embeddings) from the input text or code.
    """
    model = AutoModel.from_pretrained('microsoft/codebert-base')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state

# test_function_code --------------------

def test_extract_features():
    """
    Test the extract_features function.

    Raises:
        AssertionError: If the function does not work as expected.
    """
    input_text = 'def hello_world():\n    print("Hello, world!")'
    embeddings = extract_features(input_text)
    assert embeddings is not None, 'The function did not return any embeddings.'
    assert embeddings.size(0) == 1, 'The function did not return the correct number of embeddings.'

# call_test_function_code --------------------

test_extract_features()