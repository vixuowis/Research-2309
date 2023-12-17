# requirements_file --------------------

!pip install -U transformers==4.16.2 pandas==1.3.5

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

# function_code --------------------

def extract_financial_info(table_data, query):
    # Load the tokenizer and model from Hugging Face hub
    tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large-1024shot')
    model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large-1024shot')

    # Convert the provided table_data dictionary to a pandas DataFrame
    table = pd.DataFrame.from_dict(table_data)

    # Encode the table and the query into model inputs
    encoding = tokenizer(table=table, query=query, return_tensors='pt')

    # Generate the answer
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer

# test_function_code --------------------

def test_extract_financial_info():
    print('Testing extract_financial_info function.')
    # Sample financial table and query
    sample_data = {'year': [1990, 2000, 2010, 2020], 'revenue': [100, 200, 300, 400]}
    query = 'What was the revenue in 2010?'

    # Expected answer
    expected_answer = 'The revenue in 2010 was 300.'

    # Test case
    print('Testing case [1/1] started.')
    actual_answer = extract_financial_info(sample_data, query)
    assert actual_answer == expected_answer, f'Test case failed: Expected {expected_answer}, got {actual_answer}'
    print('Testing finished.')

# Run the test function
test_extract_financial_info()