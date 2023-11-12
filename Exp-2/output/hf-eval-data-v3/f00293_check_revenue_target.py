# function_import --------------------

from transformers import TapasForQuestionAnswering, TapasTokenizer
import pandas as pd

# function_code --------------------

def check_revenue_target(table: dict, query: str):
    """
    Check if the total revenue for last week met our target revenue.

    Args:
        table (dict): A dictionary representing the table with daily revenue for the week.
        query (str): A string representing the query asking whether the target revenue has been achieved.

    Returns:
        tuple: A tuple containing predicted answer coordinates and predicted aggregation indices.
    """
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-small-finetuned-wtq')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-small-finetuned-wtq')

    table = pd.DataFrame(table)
    inputs = tokenizer(table=table, queries=query, return_tensors='pt')
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())

    return predicted_answer_coordinates, predicted_aggregation_indices

# test_function_code --------------------

def test_check_revenue_target():
    """
    Test the function check_revenue_target.
    """
    table1 = {"Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
             "Revenue": [2000, 2500, 3000, 3500, 4000, 4500, 5000]}
    query1 = "Did the total revenue meet the target revenue of 24000?"
    result1 = check_revenue_target(table1, query1)
    assert isinstance(result1, tuple)

    table2 = {"Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
             "Revenue": [1000, 1500, 2000, 2500, 3000, 3500, 4000]}
    query2 = "Did the total revenue meet the target revenue of 18000?"
    result2 = check_revenue_target(table2, query2)
    assert isinstance(result2, tuple)

    print('All Tests Passed')

# call_test_function_code --------------------

test_check_revenue_target()