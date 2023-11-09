# function_import --------------------

from transformers import PreTrainedTokenizerFast, BartModel

# function_code --------------------

def extract_features(input_text: str) -> torch.Tensor:
    """
    Extracts features from Korean text using a pre-trained KoBART model.

    Args:
        input_text (str): The Korean text from which to extract features.

    Returns:
        torch.Tensor: The extracted features in the form of a tensor.
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    model = BartModel.from_pretrained('gogamza/kobart-base-v2')
    tokens = tokenizer(input_text, return_tensors="pt")
    features = model(**tokens)['last_hidden_state']
    return features

# test_function_code --------------------

def test_extract_features():
    """
    Tests the extract_features function by passing a sample Korean text and checking the output type.
    """
    sample_text = '한국어 텍스트'
    features = extract_features(sample_text)
    assert isinstance(features, torch.Tensor), 'The output should be a torch.Tensor.'

# call_test_function_code --------------------

test_extract_features()