# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features_from_russian_text(input_text: str) -> torch.Tensor:
    """
    Extract features from Russian text using a pre-trained RuBERT model.

    Args:
        input_text (str): A string containing Russian text.

    Returns:
        torch.Tensor: A tensor containing extracted features.

    Raises:
        ValueError: If the input_text is not a string.
    """
    if not isinstance(input_text, str):
        raise ValueError('The input text must be a string.')
    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model(**inputs)
    features = outputs.last_hidden_state
    return features

# test_function_code --------------------

def test_extract_features_from_russian_text():
    print('Testing started.')
    sample_text = 'Пример текста на русском языке.'

    # Test case 1: Correct string input
    print('Testing case [1/1] started.')
    features = extract_features_from_russian_text(sample_text)
    assert features.shape[0] == 1, f'Test case [1/1] failed: Expected the number of examples in output to be 1, got {features.shape[0]}.'
    print('Testing finished.')

# call_test_function_line --------------------

test_extract_features_from_russian_text()