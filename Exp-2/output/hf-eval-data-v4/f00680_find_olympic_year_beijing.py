# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def find_olympic_year_beijing(host_cities):
    # Load the tokenizer and model from the TAPEX-base
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base')

    # Convert the host cities to a pandas DataFrame
    table = pd.DataFrame.from_dict(host_cities)

    # Our query to find the Olympic year when Beijing was the host city
    query = 'select year where city = beijing'

    # Tokenize the table and query
    encoding = tokenizer(table=table, query=query, return_tensors='pt')

    # Use the model to generate an answer to the query
    outputs = model.generate(**encoding)

    # Decode the generated answer
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return result[0]

# test_function_code --------------------

def test_find_olympic_year_beijing():
    print('Testing find_olympic_year_beijing function.')

    # Example data set to find the Olympic year for Beijing
    host_cities = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012, 2021],
        'city': ['athens', 'paris', 'st. louis', 'athens', 'beijing', 'london', 'tokyo']
    }

    # Expected output is the year 2008
    expected_year = '2008'

    # Run the test
    result = find_olympic_year_beijing(host_cities)
    assert result == expected_year, f'Test failed: Expected {expected_year} but got {result}.'
    print('Test passed!')

# Run the test function
test_find_olympic_year_beijing()