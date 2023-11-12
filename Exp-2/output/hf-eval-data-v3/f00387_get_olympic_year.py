# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def get_olympic_year(table: pd.DataFrame, query: str) -> str:
    '''
    This function takes a pandas DataFrame and a query string as input, uses the Tapex model to answer the query based on the table data, and returns the answer as a string.

    Args:
        table (pd.DataFrame): A pandas DataFrame containing the table data.
        query (str): A string containing the query.

    Returns:
        str: The answer to the query.
    '''
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base')
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer

# test_function_code --------------------

def test_get_olympic_year():
    '''
    This function tests the get_olympic_year function by using a few test cases.
    '''
    data1 = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['Athens', 'Paris', 'St. Louis', 'Athens', 'Beijing', 'London']
    }
    table1 = pd.DataFrame.from_dict(data1)
    query1 = 'Select the year when Beijing hosted the Olympic games'
    assert get_olympic_year(table1, query1) == '2008'

    data2 = {
        'year': [1896, 1900, 1904, 2004, 2012],
        'city': ['Athens', 'Paris', 'St. Louis', 'Athens', 'London']
    }
    table2 = pd.DataFrame.from_dict(data2)
    query2 = 'Select the year when Beijing hosted the Olympic games'
    assert get_olympic_year(table2, query2) == 'Beijing did not host the Olympic games in the given years.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_olympic_year()