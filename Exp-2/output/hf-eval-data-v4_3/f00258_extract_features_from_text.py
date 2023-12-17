# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features_from_text(text: str):
    """
    Extract features from Russian language text using pre-trained model.

    Args:
        text (str): A text string in Russian language.

    Returns:
        torch.Tensor: A tensor representing extracted features from the text.

    Raises:
        ValueError: If the text is not in Russian or is empty.
    """
    if not text or not isinstance(text, str):
        raise ValueError('The text must be a non-empty string.')
    tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    features = outputs.last_hidden_state
    return features


# test_function_code --------------------

def test_extract_features_from_text():
    print("Testing started.")
    sample_text = 'Привет, как дела?'

    # Test case 1: Non-empty Russian text
    print("Testing case [1/1] started.")
    features = extract_features_from_text(sample_text)
    assert features.shape[1] == 768, f"Test case [1/1] failed: Expected feature dimension 768, got {features.shape[1]}"
    print("Testing finished.")


# call_test_function_line --------------------

test_extract_features_from_text()