# function_import --------------------

from transformers import AutoModel, AutoTokenizer

# function_code --------------------

def extract_features(text):
    """
    This function uses the pre-trained ConvBERT model 'YituTech/conv-bert-base' to extract features from the given text.
    
    Args:
        text (str): The text data from which to extract features.
    
    Returns:
        torch.Tensor: The extracted features from the last hidden state of the ConvBERT model.
    """
    conv_bert_model = AutoModel.from_pretrained('YituTech/conv-bert-base')
    tokenizer = AutoTokenizer.from_pretrained('YituTech/conv-bert-base')
    input_tokens = tokenizer.encode(text, return_tensors='pt')
    features = conv_bert_model(**input_tokens).last_hidden_state
    return features

# test_function_code --------------------

def test_extract_features():
    """
    This function tests the 'extract_features' function with a sample text.
    """
    sample_text = 'This is a sample text for testing.'
    features = extract_features(sample_text)
    assert features is not None, 'No features extracted.'
    assert features.size(0) == 1, 'Incorrect number of features extracted.'

# call_test_function_code --------------------

test_extract_features()