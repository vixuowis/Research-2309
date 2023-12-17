# requirements_file --------------------

import subprocess

requirements = ["transformers", "pandas"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd


# function_code --------------------

def find_olympic_year_beijing(data, query):
    """
    Find the year when Beijing hosted the Olympic games from the given dataset.

    Args:
        data (dict): A dictionary containing 'year' and 'city', where each key has a list of values.
        query (str): The natural language query string.

    Returns:
        str: The year when Beijing hosted the Olympic games as a string.

    Raises:
        ValueError: If Beijing is not found in the dataset.
    """
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base')
    
    table = pd.DataFrame.from_dict(data)
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    if 'Beijing' not in data['city']:
        raise ValueError('Beijing was not found in the dataset')

    return answer


# test_function_code --------------------

def test_find_olympic_year_beijing():
    print("Testing started.")
    data = {
        "year": [1896, 1900, 1904, 2004, 2008, 2012],
        "city": ["Athens", "Paris", "St. Louis", "Athens", "Beijing", "London"]
    }
    query = "Select the year when Beijing hosted the Olympic games"

    # Test case 1: Valid scenario
    print("Testing case [1/1] started.")
    year = find_olympic_year_beijing(data, query)
    assert year == '2008', f"Test case [1/1] failed: Expected '2008', got {year}"
    print("Testing finished.")


# call_test_function_line --------------------

test_find_olympic_year_beijing()