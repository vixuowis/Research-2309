# function_import --------------------

from transformers import BertTokenizer, AutoModel
import torch

# function_code --------------------

def extract_features(input_text):
    """
    This function is used to extract features from Indonesian text using the IndoBERT model.

    Args:
        input_text (str): The Indonesian text from which to extract features.

    Returns:
        torch.Tensor: The contextual representation of the input text.
    """
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    model = AutoModel.from_pretrained('indobenchmark/indobert-base-p1')
    encoded_input = tokenizer.encode(input_text, return_tensors='pt')
    contextual_representation = model(encoded_input)[0]
    return contextual_representation

# test_function_code --------------------

def test_extract_features():
    """
    This function is used to test the 'extract_features' function.
    It uses a sample Indonesian text and checks if the output is a torch.Tensor.
    """
    sample_text = 'Saya suka makan nasi goreng'
    output = extract_features(sample_text)
    assert isinstance(output, torch.Tensor), 'Output is not a torch.Tensor'

# call_test_function_code --------------------

test_extract_features()