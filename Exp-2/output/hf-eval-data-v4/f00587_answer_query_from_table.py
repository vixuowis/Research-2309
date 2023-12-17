# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def answer_query_from_table(data, query):
    """
    Given a dictionary representing table data and a query string,
    this function uses a pre-trained TAPEX model to find and return the answer.

    Args:
        data (dict): The table data in dictionary format.
        query (str): The question to be answered from the table.

    Returns:
        str: The answer to the query.
    """
    # Load the tokenizer and model
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base-finetuned-wtq')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base-finetuned-wtq')

    # Convert the data to a pandas DataFrame
    table = pd.DataFrame.from_dict(data)

    # Encode the table and query for the model input
    encoding = tokenizer(table=table, query=query, return_tensors='pt')

    # Generate the answer
    outputs = model.generate(**encoding)

    # Decode the generated ids to get the answer
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return answer

# test_function_code --------------------

def test_answer_query_from_table():
    print("Testing started.")
    # Example table data
    data = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['athens', 'paris', 'st. louis', 'athens', 'beijing', 'london']
    }
    # Test query
    query = "In which year did Beijing host the Olympic Games?"

    # Expected answer
    expected_answer = '2008'

    # Get the actual answer from the table
    answer = answer_query_from_table(data, query)
    assert answer == expected_answer, f"Test failed: Expected '{{expected_answer}}' but got '{{answer}}'"
    print("Testing finished.")

# Run the test
test_answer_query_from_table()