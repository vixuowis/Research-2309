# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def find_olympic_year_beijing(table_data):
    """
    Find the year when Beijing hosted the Olympic games using TAPEX.

    Args:
    - table_data (dict): A dictionary containing 'year' and 'city' as keys mapping to lists

    Returns:
    str: The year Beijing hosted the Olympic games represented as a string
    """
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base')
    table = pd.DataFrame.from_dict(table_data)
    query = "Select the year when Beijing hosted the Olympic games"
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer

# test_function_code --------------------

def test_find_olympic_year_beijing():
    print("Testing started.")
    test_data = {
        "year": [1896, 1900, 1904, 2004, 2008, 2012],
        "city": ["Athens", "Paris", "St. Louis", "Athens", "Beijing", "London"]
    }
    expected_answer = "2008"

    print("Testing find_olympic_year_beijing function")
    actual_answer = find_olympic_year_beijing(test_data)
    assert actual_answer == expected_answer, f"Test failed: expected {{expected_answer}}, but got {{actual_answer}}"
    print("Testing finished successfully.")

# Run the test function
test_find_olympic_year_beijing()