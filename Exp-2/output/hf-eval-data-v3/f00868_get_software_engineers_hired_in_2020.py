# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def get_software_engineers_hired_in_2020(employee_data):
    """
    This function finds all employees with the title of "Software Engineer" hired in 2020.

    Args:
        employee_data (dict): A dictionary containing employee data. The keys are column names and the values are lists of column values.

    Returns:
        list: A list of employees who are software engineers and were hired in 2020.
    """
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-large-sql-execution')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-large-sql-execution')

    table = pd.DataFrame.from_dict(employee_data)

    query = 'SELECT * FROM table WHERE title = "Software Engineer" AND hire_date >= "2020-01-01" AND hire_date <= "2020-12-31"'

    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return results

# test_function_code --------------------

def test_get_software_engineers_hired_in_2020():
    """
    This function tests the get_software_engineers_hired_in_2020 function.
    """
    employee_data = {
        'name': ['John Doe', 'Jane Doe', 'Bob Smith'],
        'title': ['Software Engineer', 'Data Scientist', 'Software Engineer'],
        'department': ['Engineering', 'Data Science', 'Engineering'],
        'hire_date': ['2020-01-01', '2019-01-01', '2020-01-01']
    }
    assert len(get_software_engineers_hired_in_2020(employee_data)) == 2

    employee_data = {
        'name': ['John Doe', 'Jane Doe', 'Bob Smith'],
        'title': ['Data Scientist', 'Data Scientist', 'Data Scientist'],
        'department': ['Data Science', 'Data Science', 'Data Science'],
        'hire_date': ['2020-01-01', '2019-01-01', '2020-01-01']
    }
    assert len(get_software_engineers_hired_in_2020(employee_data)) == 0

    employee_data = {
        'name': ['John Doe', 'Jane Doe', 'Bob Smith'],
        'title': ['Software Engineer', 'Software Engineer', 'Software Engineer'],
        'department': ['Engineering', 'Engineering', 'Engineering'],
        'hire_date': ['2019-01-01', '2019-01-01', '2019-01-01']
    }
    assert len(get_software_engineers_hired_in_2020(employee_data)) == 0

    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_software_engineers_hired_in_2020()