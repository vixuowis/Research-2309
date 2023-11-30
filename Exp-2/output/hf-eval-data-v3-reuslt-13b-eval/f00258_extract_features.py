# function_import --------------------

import torch
from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features(text):
    """
    Extract features from the given text using the 'DeepPavlov/rubert-base-cased' model.

    Args:
        text (str): The text from which to extract features.

    Returns:
        torch.Tensor: The extracted features.
    """
    
    # get model and tokenizer
    model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')
    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    
    # encode text and get features
    encoded = tokenizer(text, return_tensors='pt', padding=True)
    with torch.no_grad():
        features = model(**encoded).pooler_output[0]
        
    return features

# test_function_code --------------------

def test_extract_features():
    """
    Test the extract_features function.
    """
    text = 'Введите текст на русском языке здесь'
    features = extract_features(text)
    assert features is not None
    assert features.size(0) == 1

    text = 'Это еще один тестовый текст'
    features = extract_features(text)
    assert features is not None
    assert features.size(0) == 1

    text = 'И это последний тестовый текст'
    features = extract_features(text)
    assert features is not None
    assert features.size(0) == 1

    return 'All Tests Passed'


# call_test_function_code --------------------

test_extract_features()