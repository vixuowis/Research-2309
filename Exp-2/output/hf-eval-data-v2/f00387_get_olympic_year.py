# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def get_olympic_year(table: pd.DataFrame, query: str) -> str:
    """
    This function uses the pre-trained model 'microsoft/tapex-base' to answer the query regarding historical Olympic host cities.

    Args:
        table (pd.DataFrame): A dataframe containing the years and cities of the Olympic games.
        query (str): A string containing the query to be answered.

    Returns:
        str: The year when Beijing hosted the Olympic games.
    """
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base')
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer

# test_function_code --------------------

def test_get_olympic_year():
    """
    This function tests the 'get_olympic_year' function with a sample dataset.
    """
    data = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['Athens', 'Paris', 'St. Louis', 'Athens', 'Beijing', 'London']
    }
    table = pd.DataFrame.from_dict(data)
    query = 'Select the year when Beijing hosted the Olympic games'
    assert get_olympic_year(table, query) == '2008'

# call_test_function_code --------------------

test_get_olympic_year()