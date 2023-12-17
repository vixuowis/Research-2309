# requirements_file --------------------

import subprocess

requirements = ["transformers", "pandas"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def table_question_answering(csv_file: str, query: str) -> str:
    """
    Answers a given query based on the data from a CSV file using a table-based QA model.

    Args:
        csv_file (str): The file path to the CSV containing the tabular data.
        query (str): The natural language question to be answered.

    Returns:
        str: The answer to the query.

    Raises:
        FileNotFoundError: If the csv_file does not exist.
        ValueError: If the query is not provided or empty.
    """
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large-1024shot')
    model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large-1024shot')

    # Load the table from the CSV file
    try:
        table = pd.read_csv(csv_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    if not query:
        raise ValueError("Query must not be empty.")

    # Encode the table and query
    encoding = tokenizer(table=table, query=query, return_tensors='pt')

    # Generate and decode the answer
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return answer


# test_function_code --------------------

def test_table_question_answering():
    print("Testing started.")

    # Test data sample
    data_sample = {
        'Year': [2016, 2020],
        'Host_City': ['Rio', 'Tokyo']
    }
    query_sample = "In which year did Tokyo host the Olympics?"

    # Create a temporary CSV file for testing
    test_csv = 'temp_test.csv'
    pd.DataFrame.from_dict(data_sample).to_csv(test_csv, index=False)

    # Test case 1: Valid CSV and query
    print("Testing case [1/2] started.")
    answer = table_question_answering(test_csv, query_sample)
    assert answer == "2020", f"Test case [1/2] failed: Expected answer '2020', got {answer}"

    # Test case 2: Invalid CSV
    print("Testing case [2/2] started.")
    try:
        table_question_answering('nonexistent.csv', query_sample)
        assert False, "Test case [2/2] failed: Expected FileNotFoundError"
    except FileNotFoundError:
        pass  # Expected

    print("Testing finished.")


# call_test_function_line --------------------

test_table_question_answering()