# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def find_highest_market_cap(table, query):
    """
    Finds the company with the highest market cap in the given table using NLP Table Question Answering.

    Args:
        table (dict): The table containing the Korean stock market data in the form of a
                      dictionary with 'header' and 'rows' as keys.
        query (str): The query asking which company has a higher market cap.

    Returns:
        str: The company with the highest market cap according to the table data.

    Raises:
        ValueError: If the table format is invalid.
    """
    # Ensure table has required structure
    if 'header' not in table or 'rows' not in table:
        raise ValueError("Invalid table format.")

    # Load the table-question-answering pipeline
    table_qa = pipeline('table-question-answering', model='dsba-lab/koreapas-finetuned-korwikitq')

    # Get the answer to the query
    answer = table_qa(table=table, query=query)

    # Return the company with the highest market cap
    return answer['answer']

# test_function_code --------------------

def test_find_highest_market_cap():
    print("Testing started.")

    # Example table data
    table = {
        'header': ['company', 'stock price', 'market cap'],
        'rows': [['samsung', 50000, 100000], ['lg', 30000, 45000]]
    }

    # Test case 1: Valid table data
    print("Testing case [1/1] started.")
    expected_result = 'samsung'
    actual_result = find_highest_market_cap(table, 'Which company has a higher market cap?')
    assert expected_result == actual_result, f"Test case [1/1] failed: Expected {expected_result}, got {actual_result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_find_highest_market_cap()