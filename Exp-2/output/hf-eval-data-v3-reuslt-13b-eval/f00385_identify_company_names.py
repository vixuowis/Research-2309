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
    # Check if the given text is of type str. Else raise an error.
    if not isinstance(text,str):
        raise ValueError('Please provide a string as input to this function.') 
    
    # Initialize the tokenizer for the desired transformer model from Hugging Face Transformers. In our case, we choose BERT-base.
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english") 
    
    # Convert the input text to a tensor for further processing.
    inputs = tokenizer(text, return_tensors="pt")  
    
    # Initialize the model from Hugging Face Transformers. In our case, we choose BERT-base.
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english") 
    
    # Set the model in evaluation mode.
    model.eval()  
    
    # Make the predictions for named entity recognition on the input text and return the outputs.
    with torch.no_grad():
        outputs = model(**inputs)
    
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