# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

# function_code --------------------

def get_total_sales(sales_data_table: pd.DataFrame, question: str) -> int:
    """
    Get total sales of a specific product based on a table containing sales information per week.

    Args:
        sales_data_table (pd.DataFrame): A DataFrame containing sales data.
        question (str): A question related to the sales data.

    Returns:
        int: The total sales of the specific product.
    """
    tokenizer = TapasTokenizer.from_pretrained('lysandre/tapas-temporary-repo')
    model = TapasForQuestionAnswering.from_pretrained('lysandre/tapas-temporary-repo')
    inputs = tokenizer(table=sales_data_table, queries=question, return_tensors='pt')
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())
    return predicted_answer_coordinates, predicted_aggregation_indices

# test_function_code --------------------

def test_get_total_sales():
    """
    Test the function get_total_sales.
    """
    sales_data_table = pd.DataFrame({'Product': ['A', 'B', 'C'], 'Sales': [100, 200, 300], 'Week': ['1', '1', '1']})
    question = 'What is the total sales of product A in week 1?'
    assert isinstance(get_total_sales(sales_data_table, question), int)
    sales_data_table = pd.DataFrame({'Product': ['A', 'B', 'C'], 'Sales': [100, 200, 300], 'Week': ['2', '2', '2']})
    question = 'What is the total sales of product B in week 2?'
    assert isinstance(get_total_sales(sales_data_table, question), int)
    sales_data_table = pd.DataFrame({'Product': ['A', 'B', 'C'], 'Sales': [100, 200, 300], 'Week': ['3', '3', '3']})
    question = 'What is the total sales of product C in week 3?'
    assert isinstance(get_total_sales(sales_data_table, question), int)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_total_sales()