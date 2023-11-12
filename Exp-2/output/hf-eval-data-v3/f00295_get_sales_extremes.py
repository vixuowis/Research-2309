# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

# function_code --------------------

def get_sales_extremes(sales_data_table: pd.DataFrame, query: str = 'What are the highest and lowest sales numbers?') -> tuple:
    """
    This function uses the TapasForQuestionAnswering model from Hugging Face Transformers to answer a query based on a given sales data table.
    The model is designed to answer questions based on a given data table.
    The function returns the highest and lowest sales numbers as an aggregated answer.

    Args:
        sales_data_table (pd.DataFrame): The sales data table.
        query (str, optional): The query to be answered. Defaults to 'What are the highest and lowest sales numbers?'.

    Returns:
        tuple: The highest and lowest sales numbers.
    """
    tokenizer = TapasTokenizer.from_pretrained('lysandre/tapas-temporary-repo')
    model = TapasForQuestionAnswering.from_pretrained('lysandre/tapas-temporary-repo')
    inputs = tokenizer(table=sales_data_table, queries=query, return_tensors='pt')
    outputs = model(**inputs)
    predicted_answer_coordinates, _ = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())
    highest_sales, lowest_sales = extract_aggregated_sales(predicted_answer_coordinates)
    return highest_sales, lowest_sales

# test_function_code --------------------

def test_get_sales_extremes():
    """
    This function tests the get_sales_extremes function.
    It uses a sample sales data table and a query to test the function.
    The function asserts if the returned highest and lowest sales numbers are as expected.
    """
    sales_data_table = pd.DataFrame({'Product': ['A', 'B', 'C', 'D', 'E'], 'Sales': [100, 200, 300, 400, 500]})
    highest_sales, lowest_sales = get_sales_extremes(sales_data_table)
    assert highest_sales == 500, 'Test Case 1 Failed'
    assert lowest_sales == 100, 'Test Case 2 Failed'
    print('All Test Cases Passed')

# call_test_function_code --------------------

test_get_sales_extremes()