# requirements_file --------------------

import subprocess

requirements = ["transformers", "pandas"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def answer_query_using_table_data(table_data, query):
    """Answer a query using the provided table data.

    Args:
        table_data (dict): A dictionary containing the table data, where keys are column names.
        query (str): The question/query to be answered using the table data.

    Returns:
        str: The answer to the query as determined from the table data.

    Raises:
        ValueError: If the provided table_data is empty or query is None.
    """
    if not table_data or query is None:
        raise ValueError('The table data must not be empty and the query must not be None.')

    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base-finetuned-wtq')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base-finetuned-wtq')
    table = pd.DataFrame.from_dict(table_data)
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer[0]

# test_function_code --------------------

def test_answer_query_using_table_data():
    print("Testing started.")
    # Sample table data and query for testing
    data = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['athens', 'paris', 'st. louis', 'athens', 'beijing', 'london']
    }
    query = "In which year did beijing host the Olympic Games?"

    # Testing case 1
    print("Testing case [1/1] started.")
    answer = answer_query_using_table_data(data, query)
    assert answer == '2008', f"Test case [1/1] failed: Expected '2008', got {answer}"
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_query_using_table_data()