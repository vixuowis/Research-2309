# function_import --------------------

import torch
from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features_from_russian_text(input_text: str):
    """
    Extract features from Russian text using the pre-trained model 'DeepPavlov/rubert-base-cased'.

    Args:
        input_text (str): The input Russian text from which to extract features.

    Returns:
        torch.Tensor: The extracted features from the input text.
    """
    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model(**inputs)
    features = outputs.last_hidden_state
    return features

# test_function_code --------------------

def test_extract_features_from_russian_text():
    """
    Test the function 'extract_features_from_russian_text'.
    """
    input_text = 'Пример текста на русском языке.'
    features = extract_features_from_russian_text(input_text)
    assert features is not None, 'The extracted features should not be None.'
    assert features.size(0) == 1, 'The size of the first dimension of the extracted features should be 1.'
    assert features.size(1) == len(input_text.split()), 'The size of the second dimension of the extracted features should be equal to the number of words in the input text.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_features_from_russian_text()