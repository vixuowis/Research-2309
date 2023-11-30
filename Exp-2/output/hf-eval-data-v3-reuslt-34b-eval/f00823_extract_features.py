# function_import --------------------

from transformers import BertTokenizer, AutoModel
import torch

# function_code --------------------

def extract_features(input_text):
    """
    Extracts contextual representation of the input text using IndoBERT model.

    Args:
        input_text (str): The input text in Indonesian language.

    Returns:
        torch.Tensor: The contextual representation of the input text.

    Raises:
        OSError: If there is a problem in loading the pretrained model or tokenizer.
    """
    try:
        tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
        model = AutoModel.from_pretrained("indobenchmark/indobert-base-p2")
        
        # Prepare input
        text = input_text
        tokens = tokenizer(text, 
                           max_length=512, 
                           padding='longest', 
                           truncation=True, 
                           return_tensors='pt')
        
        # Extract features
        with torch.no_grad():
            features = model(**tokens).last_hidden_state[0][0].numpy()
    except OSError as e:
        raise Exception("Problem loading the pretrained models/tokenizers") from e
    return features

# test_function_code --------------------

def test_extract_features():
    """
    Tests the function 'extract_features'.
    """
    sample_text = 'Saya suka makan nasi goreng'
    output = extract_features(sample_text)
    assert isinstance(output, torch.Tensor), 'Output is not a torch.Tensor'
    assert output.shape[0] == 1, 'Output shape is not correct'
    print('All Tests Passed')


# call_test_function_code --------------------

test_extract_features()