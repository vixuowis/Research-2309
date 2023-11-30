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
        
        # Initialize IndoBERT Model and Tokenizer
        
        print("[INFO] Loading Pre-Trained Models")
        tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        model = AutoModel.from_pretrained('indobenchmark/indobert-base-p1') # or indobenchmark/indobert-large-p2

        input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0) 
        
        # Extract Features using IndoBERT model (Encoder)
        
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0] # [batch, seq, dim]
            
    except OSError as err:
        print("[ERROR] Cannot load the pre-trained models.")
        raise
    
    return last_hidden_states.squeeze()

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