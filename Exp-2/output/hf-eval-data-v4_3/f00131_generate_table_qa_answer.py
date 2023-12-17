# requirements_file --------------------

import subprocess

requirements = ["transformers", "pandas"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

# function_code --------------------

def generate_table_qa_answer(table_data, query):
    """
    Generates an answer to a natural language question based on table data.

    Args:
        table_data (dict): A dictionary representing the table data.
        query (str): The natural language question to be answered.

    Returns:
        str: The predicted answer to the question.

    Raises:
        ValueError: If the table data is not a dictionary or the query is not a string.
    """
    # Validate the input arguments
    if not isinstance(table_data, dict):
        raise ValueError("The table_data argument must be a dictionary.")
    if not isinstance(query, str):
        raise ValueError("The query argument must be a string.")

    # Initialize the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large-1024shot')
    model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large-1024shot')

    # Convert table data into a pandas DataFrame
    table = pd.DataFrame.from_dict(table_data)

    # Tokenize the table and the query for the model
    encoding = tokenizer(table=table, query=query, return_tensors='pt')

    # Generate the answer
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer[0]

# test_function_code --------------------

def test_generate_table_qa_answer():
    print("Testing started.")
    # Define sample table data
    sample_table_data = {'year': [1896, 1900, 1904, 2004, 2008, 2012], 'city': ['Athens', 'Paris', 'St. Louis', 'Athens', 'Beijing', 'London']}

    # Test case 1: Check for a valid question answer pair
    print("Testing case [1/2] started.")
    query1 = "In which year did Beijing host the Olympic Games?"
    expected_answer1 = "2008"
    answer1 = generate_table_qa_answer(sample_table_data, query1)
    assert answer1 == expected_answer1, f"Test case [1/2] failed: Expected {expected_answer1}, got {answer1}"

    # Test case 2: Check for proper error handling
    print("Testing case [2/2] started.")
    invalid_table_data = 'not a dict'
    query2 = "In which year did London host the Olympic Games?"
    try:
        generate_table_qa_answer(invalid_table_data, query2)
        assert False, "Test case [2/2] failed: ValueError expected but not raised."
    except ValueError as e:
        assert str(e) == "The table_data argument must be a dictionary.", f"Test case [2/2] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_table_qa_answer()