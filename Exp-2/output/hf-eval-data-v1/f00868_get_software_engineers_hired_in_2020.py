from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

def get_software_engineers_hired_in_2020(employee_data):
    """
    This function finds all employees with the title of "Software Engineer" hired in 2020.

    Args:
        employee_data (dict): A dictionary containing the company's employee data.

    Returns:
        list: A list of employees who are software engineers and were hired in 2020.
    """
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-large-sql-execution')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-large-sql-execution')

    table = pd.DataFrame.from_dict(employee_data)

    query = 'SELECT * FROM table WHERE title = "Software Engineer" AND hire_date >= "2020-01-01" AND hire_date <= "2020-12-31"'

    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)