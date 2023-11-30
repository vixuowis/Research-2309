# function_import --------------------

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# function_code --------------------

def identify_company_names(text):
    """
    Identify company names from a given text using a pre-trained model from Hugging Face Transformers.

    Args:
        text (str): The input text from which company names are to be identified.

    Returns:
        outputs (torch.Tensor): The model outputs, which include the predicted token classifications.

    Raises:
        ValueError: If the input text is not a string.
    """
    
    # Check if input text is valid
    if not isinstance(text, str):
        raise ValueError("Input must be provided as a string.")
      
    else:
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("Davlan/distilbert-base-multilingual-cased-masakhaner")
        model = AutoModelForTokenClassification.from_pretrained("Davlan/distilbert-base-multilingual-cased-masakhaner")
        
        # Tokenize the text and run the model inference
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens.to(device)
    
        outputs = model(**tokens).logits[0]
        
    return outputs

# test_function_code --------------------

def test_identify_company_names():
    """
    Test the identify_company_names function with various test cases.
    """
    # Test with a simple text
    text = 'Apple Inc. is an American multinational technology company.'
    outputs = identify_company_names(text)
    assert outputs is not None, 'The output should not be None.'

    # Test with a text that does not contain any company names
    text = 'This is a test sentence without any company names.'
    outputs = identify_company_names(text)
    assert outputs is not None, 'The output should not be None.'

    # Test with a text that contains multiple company names
    text = 'Apple and Microsoft are two of the biggest technology companies in the world.'
    outputs = identify_company_names(text)
    assert outputs is not None, 'The output should not be None.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_identify_company_names()