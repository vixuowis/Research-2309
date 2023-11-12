# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def get_olympic_year(city_name):
    """
    This function returns the year when the Olympic Games were held in the given city.

    Args:
        city_name (str): The name of the city.

    Returns:
        str: The year when the Olympic Games were held in the given city.
    """
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base')
    data = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['athens', 'paris', 'st. louis', 'athens', 'beijing', 'london']
    }
    table = pd.DataFrame.from_dict(data)
    query = f"select year where city = {city_name}"
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# test_function_code --------------------

def test_get_olympic_year():
    assert get_olympic_year('beijing') == '2008'
    assert get_olympic_year('athens') in ['1896', '2004']
    assert get_olympic_year('london') == '2012'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_get_olympic_year())