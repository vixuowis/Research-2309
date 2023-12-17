# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def find_employees_hired_in_2020(employee_data):
    """
    Finds all employees with the title 'Software Engineer' hired in 2020.

    Args:
        employee_data (dict): A dictionary containing employees' names, titles, departments, and hire dates.

    Returns:
        list: A list of dictionaries containing the information of the employees that match the criteria.

    Raises:
        Exception: If the model fails to generate the results.
    """
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-large-sql-execution')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-large-sql-execution')
    table = pd.DataFrame.from_dict(employee_data)
    query = 'SELECT * FROM table WHERE title = "Software Engineer" AND hire_date >= "2020-01-01" AND hire_date <= "2020-12-31"'
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if not results or 'error' in results[0].lower():
        raise Exception('Model failed to generate the results')
    return results

# test_function_code --------------------

def test_find_employees_hired_in_2020():
    print("Testing started.")
    # Prepare a sample employee data
    sample_data = {
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'title': ['Software Engineer', 'Software Engineer', 'Manager', 'Software Engineer'],
        'department': ['Engineering', 'Engineering', 'Marketing', 'Engineering'],
        'hire_date': ['2020-02-01', '2019-05-15', '2020-07-23', '2020-06-01']
    }

    # Testing case 1
    print("Testing case [1/1] started.")
    expected_result = [...] # The expected result based on the sample data
    result = find_employees_hired_in_2020(sample_data)
    assert result == expected_result, f"Test case [1/1] failed: {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_find_employees_hired_in_2020()