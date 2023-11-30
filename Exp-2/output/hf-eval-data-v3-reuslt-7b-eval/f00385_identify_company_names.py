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
    
    # check if text is of correct type and return error message if not
    if type(text) != str:
        raise TypeError("Text needs to be provided as a string.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pretrained_model = 'dslim/bert-base-NER' # load the pretrained model from Hugging Face Transformers
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    pretrained_weights = torch.load('models/bert-ner.pt', map_location=device)
    
    tokens = tokenizer.tokenize(text) # split text into tokens
    inputs = tokenizer(text, return_tensors='pt')  # add special tokens takes care of all the preprocessing
    outputs = model(**inputs) # get predictions
    
    return outputs[0] # return the predicted classes

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