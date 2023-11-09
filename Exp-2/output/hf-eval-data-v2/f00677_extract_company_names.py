# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification

# function_code --------------------

def extract_company_names(review):
    """
    Extracts company names from a given review using a pre-trained model.

    Args:
        review (str): The review from which to extract company names.

    Returns:
        list: A list of company names found in the review.
    """
    model = AutoModelForTokenClassification.from_pretrained('ismail-lucifer011/autotrain-company_all-903429548', use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained('ismail-lucifer011/autotrain-company_all-903429548', use_auth_token=True)
    inputs = tokenizer(review, return_tensors='pt')
    outputs = model(**inputs)
    company_entities = []
    for output in outputs:
        if output['entity'] == 'company':
            company_entities.append(output['word'])
    return company_entities

# test_function_code --------------------

def test_extract_company_names():
    """
    Tests the extract_company_names function.
    """
    review = 'I love AutoTrain'
    expected_output = ['AutoTrain']
    assert set(extract_company_names(review)) == set(expected_output), 'Test failed!'
    print('Test passed.')

# call_test_function_code --------------------

test_extract_company_names()