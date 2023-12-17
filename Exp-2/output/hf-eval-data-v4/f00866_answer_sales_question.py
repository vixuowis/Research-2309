# requirements_file --------------------

!pip install -U transformers==4.12.0

# function_import --------------------

from transformers import TapasForQuestionAnswering, TapasTokenizer

# function_code --------------------

def answer_sales_question(table_data, query):
    """
    Answers a question about salesperson performance data using the TAPAS model.
    
    Parameters:
    table_data (list of dict): The sales data, where each dict represents a row in the table.
    query (str): The question to be answered about the table.

    Returns:
    str: The answer to the query based on the table data.
    """
    # Load the TAPAS model and tokenizer
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wtq')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')

    # Tokenize the table and query
    inputs = tokenizer(table=table_data, queries=query, return_tensors='pt')
    # Perform inference
    outputs = model(**inputs)

    # Decoding the predicted answer
    predicted_answer_coordinates = outputs.predicted_answer_coordinates
    table = {column: [row[column] for row in table_data] for column in table_data[0]}
    answer = tokenizer.decode_answer(table=table, coordinates=predicted_answer_coordinates[0])

    return answer

# test_function_code --------------------

def test_answer_sales_question():
    print("Testing started.")
    # Sample sales data
    sales_data = [
        {'Region': 'East', 'Month': 'January', 'Sales': '1000'},
        {'Region': 'West', 'Month': 'February', 'Sales': '1500'}
    ]
    # Expected answer (e.g., What was the total sales in February?)
    expected_answer = '1500'

    # Test case: Correct answer retrieval
    print("Testing case [1/1] started.")
    actual_answer = answer_sales_question(sales_data, 'What was the total sales in February?')
    assert actual_answer == expected_answer, f"Test case failed: Expected {expected_answer}, but got {actual_answer}"
    print("Testing finished.")

# Run the test function
if __name__ == '__main__':
    test_answer_sales_question()