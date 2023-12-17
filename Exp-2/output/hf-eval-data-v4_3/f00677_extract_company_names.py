# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# function_code --------------------

def extract_company_names(review_text, model_name='ismail-lucifer011/autotrain-company_all-903429548', use_auth_token=True):
    """
    Extracts company names from a given text using a pre-trained NLP token classification model.

    Args:
        review_text (str): The text from which to extract company names.
        model_name (str, optional): Hugging Face model identifier. Defaults to 'ismail-lucifer011/autotrain-company_all-903429548'.
        use_auth_token (bool, optional): If True, use authentication token for the Hugging Face API. Defaults to True.

    Returns:
        list: A list containing the extracted company names.

    Raises:
        ValueError: If the review text is empty or not a string.
    """
    if not review_text or not isinstance(review_text, str):
        raise ValueError('The review text must be a non-empty string.')
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
    model = AutoModelForTokenClassification.from_pretrained(model_name, use_auth_token=use_auth_token)

    # Tokenize input and predict
    inputs = tokenizer(review_text, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

    # Extract company names
    tokens = inputs.tokens()
    company_names = []
    for token_id, label_id in zip(tokens, predictions.squeeze().tolist()):
        if label_id == 1:  # Assuming '1' corresponds to company entity
            company_names.append(tokenizer.decode(token_id))
    
    return company_names

# test_function_code --------------------

def test_extract_company_names():
    print('Testing started.')
    # Case 1: Review with known company names
    review_text_1 = 'I love AutoTrain and the services provided by Hugging Face.'
    print('Testing case [1/2] started.')
    expected_1 = ['AutoTrain', 'Hugging Face']
    result_1 = extract_company_names(review_text_1)
    assert set(result_1) == set(expected_1), f'Test case [1/2] failed: Expected {expected_1}, got {result_1}'
    
    # Case 2: Review without company names
    review_text_2 = 'This new update is amazing and improves productivity.'
    print('Testing case [2/2] started.')
    expected_2 = []
    result_2 = extract_company_names(review_text_2)
    assert set(result_2) == set(expected_2), f'Test case [2/2] failed: Expected {expected_2}, got {result_2}'
    
    print('Testing finished.')

# call_test_function_line --------------------

test_extract_company_names()