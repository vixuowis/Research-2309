# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features(text, model_name='YituTech/conv-bert-base'):
    """Extract features from text using a pre-trained ConvBERT model.

    Args:
        text (str): The text to process and extract features from.
        model_name (str): The model identifier on Hugging Face Transformers library.

    Returns:
        torch.Tensor: The extracted features as a tensor.

    Raises:
        ValueError: If the text is empty.
    """
    if not text:
        raise ValueError('Input text cannot be empty.')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    input_tokens = tokenizer.encode(text, return_tensors='pt')
    features = model(**input_tokens).last_hidden_state
    return features


# test_function_code --------------------

def test_extract_features():
    print("Testing started.")
    sample_texts = ['Hello, world!', 'The quick brown fox jumps over the lazy dog', '']

    # Test case 1: Non-empty text
    print("Testing case [1/3] started.")
    features = extract_features(sample_texts[0])
    assert features is not None, f"Test case [1/3] failed: Expected features to be non-empty tensor, got {features}"

    # Test case 2: Text with longer sentence
    print("Testing case [2/3] started.")
    features = extract_features(sample_texts[1])
    assert features.shape[1] > 1, f"Test case [2/3] failed: Expected feature shape on dim 1 to be greater than 1, got {features.shape}"

    # Test case 3: Empty text
    print("Testing case [3/3] started.")
    try:
        features = extract_features(sample_texts[2])
        assert False, "Test case [3/3] failed: Expected ValueError for empty text"
    except ValueError as e:
        assert str(e) == 'Input text cannot be empty.', f"Test case [3/3] failed: {e}"
    print("Testing finished.")


# call_test_function_line --------------------

test_extract_features()