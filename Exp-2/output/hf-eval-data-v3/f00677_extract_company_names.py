# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List

# function_code --------------------

def extract_company_names(review: str) -> List[str]:
    """
    Extract company names from a given review using a pre-trained model.

    Args:
        review (str): The review from which to extract company names.

    Returns:
        List[str]: A list of company names found in the review.

    Raises:
        TypeError: If the review is not a string.
    """
    model = AutoModelForTokenClassification.from_pretrained('ismail-lucifer011/autotrain-company_all-903429548')
    tokenizer = AutoTokenizer.from_pretrained('ismail-lucifer011/autotrain-company_all-903429548')
    inputs = tokenizer(review, return_tensors='pt')
    outputs = model(**inputs)
    company_entities = []
    for output in outputs:
        if output['entity'] == 'company':
            company_entities.append(output['word'])
    return company_entities

# test_function_code --------------------

def test_extract_company_names():
    """Tests for the `extract_company_names` function"""
    assert set(extract_company_names('I love AutoTrain')) == set(['AutoTrain']), 'Test Case 1 Failed'
    assert set(extract_company_names('Microsoft is a great company')) == set(['Microsoft']), 'Test Case 2 Failed'
    assert set(extract_company_names('Google and Apple are competitors')) == set(['Google', 'Apple']), 'Test Case 3 Failed'
    print('All Test Cases Passed')

# call_test_function_code --------------------

test_extract_company_names()