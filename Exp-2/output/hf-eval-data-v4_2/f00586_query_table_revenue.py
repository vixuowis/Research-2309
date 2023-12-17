# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import TapasForQuestionAnswering

# function_code --------------------

def query_table_revenue(table_data, year):
    """
    Query the revenue of a company from table data for a specified year.

    Args:
        table_data (list of dict): The tabular data containing 'Year' and 'Revenue' columns.
        year (str): The year for which the revenue is to be queried.

    Returns:
        str: The revenue of the company for the specified year.

    Raises:
        ValueError: If the specified year is not found in the table data.
    """
    question = f"What was the revenue of the company in {year}?"
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wikisql-supervised')
    # Assuming that the model has a method 'predict' that takes question and table_data as arguments
    prediction = model.predict(question, table_data)
    answer = prediction.answers[0] if prediction.answers else 'No data'
    return answer

# test_function_code --------------------

def test_query_table_revenue():
    print("Testing started.")
    table_data = [
      {"Year": "2018", "Revenue": "$20M"},
      {"Year": "2019", "Revenue": "$25M"},
      {"Year": "2020", "Revenue": "$30M"},
    ]

    # Testing case 1: Query for 2020
    print("Testing case [1/3] started.")
    assert query_table_revenue(table_data, '2020') == "$30M", f"Test case [1/3] failed: Expected $30M"

    # Testing case 2: Query for 2019
    print("Testing case [2/3] started.")
    assert query_table_revenue(table_data, '2019') == "$25M", f"Test case [2/3] failed: Expected $25M"

    # Testing case 3: Query for non-existing year
    print("Testing case [3/3] started.")
    try:
        query_table_revenue(table_data, '2021')
        assert False, "Test case [3/3] failed: Expected ValueError for non-existing year"
    except ValueError as e:
        assert str(e) == "Specified year not found in the table data", "Test case [3/3] failed: Incorrect ValueError message"

    print("Testing finished.")

# call_test_function_line --------------------

test_query_table_revenue()