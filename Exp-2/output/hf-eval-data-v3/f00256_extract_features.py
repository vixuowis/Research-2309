# function_import --------------------

from transformers import PreTrainedTokenizerFast, BartModel
import torch

# function_code --------------------

def extract_features(input_text: str) -> torch.Tensor:
    """
    Extract features from input text using KoBART model.

    Args:
        input_text (str): The input text in Korean.

    Returns:
        torch.Tensor: The extracted features from the input text.
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    model = BartModel.from_pretrained('gogamza/kobart-base-v2')
    tokens = tokenizer(input_text, return_tensors="pt")
    features = model(**tokens)['last_hidden_state']
    return features

# test_function_code --------------------

def test_extract_features():
    """
    Test the function extract_features.
    """
    input_text = "한국어 텍스트"
    features = extract_features(input_text)
    assert isinstance(features, torch.Tensor), 'The result is not a torch.Tensor.'
    assert features.shape[0] == 1, 'The shape of the tensor is not correct.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_extract_features()