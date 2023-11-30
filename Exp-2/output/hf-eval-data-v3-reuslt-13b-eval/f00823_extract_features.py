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
        # Loading pretrained BertTokenizer and IndoBERT model using HuggingFace Transformers library.
        indobert_tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        
        indobert_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
    except OSError as error:
        print(f"OSErrro: {error}")
    
    # Converting text to token id.
    input_text_tokenized = indobert_tokenizer([input_text], return_tensors="pt", truncation=True, max_length=512)["input_ids"]

    # Getting features from IndoBERT model.
    with torch.no_grad():
        output = indobert_model(input_text_tokenized)[0]
    
    return (output[0, 0], input_text_tokenized)

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