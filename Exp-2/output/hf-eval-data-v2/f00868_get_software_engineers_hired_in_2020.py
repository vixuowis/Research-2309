# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def get_software_engineers_hired_in_2020(employee_data):
    """
    This function finds all employees with the title of "Software Engineer" hired in 2020 from a given employee data table.

    Args:
        employee_data (dict): The employee data table as a dictionary.

    Returns:
        list: A list of employees with the title of "Software Engineer" hired in 2020.
    """
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-large-sql-execution')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-large-sql-execution')

    table = pd.DataFrame.from_dict(employee_data)

    query = 'SELECT * FROM table WHERE title = "Software Engineer" AND hire_date >= "2020-01-01" AND hire_date <= "2020-12-31"'

    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# test_function_code --------------------

def test_get_software_engineers_hired_in_2020():
    """
    This function tests the get_software_engineers_hired_in_2020 function.
    """
    employee_data = {
        'name': ['John Doe', 'Jane Doe', 'Alice', 'Bob'],
        'title': ['Software Engineer', 'Software Engineer', 'Data Scientist', 'Product Manager'],
        'department': ['Engineering', 'Engineering', 'Data', 'Product'],
        'hire_date': ['2020-01-01', '2019-01-01', '2020-01-01', '2021-01-01']
    }
    assert len(get_software_engineers_hired_in_2020(employee_data)) == 1

# call_test_function_code --------------------

test_get_software_engineers_hired_in_2020()