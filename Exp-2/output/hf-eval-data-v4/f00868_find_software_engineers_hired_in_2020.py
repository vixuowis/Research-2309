# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def find_software_engineers_hired_in_2020(employee_data):
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-large-sql-execution')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-large-sql-execution')

    table = pd.DataFrame.from_dict(employee_data)

    query = 'SELECT * FROM table WHERE title = "Software Engineer" AND hire_date >= "2020-01-01" AND hire_date <= "2020-12-31"'

    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return results

# test_function_code --------------------

def test_find_software_engineers_hired_in_2020():
    print("Testing started.")
    employee_data = [
        {'name': 'Alice', 'title': 'Software Engineer', 'department': 'Engineering', 'hire_date': '2020-05-15'},
        {'name': 'Bob', 'title': 'Project Manager', 'department': 'Engineering', 'hire_date': '2019-04-23'},
        {'name': 'Carol', 'title': 'Software Engineer', 'department': 'Engineering', 'hire_date': '2020-08-19'},
        {'name': 'Dave', 'title': 'Software Engineer', 'department': 'Engineering', 'hire_date': '2018-11-30'}
    ]

    expected_result = ['Alice', 'Carol']
    actual_result = find_software_engineers_hired_in_2020(employee_data)

    assert actual_result == expected_result, f"Test failed: Expected {expected_result}, but got {actual_result}"
    print("Testing finished.")