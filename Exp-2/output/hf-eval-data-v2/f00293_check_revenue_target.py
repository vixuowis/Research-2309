# function_import --------------------

from transformers import TapasForQuestionAnswering, TapasTokenizer

# function_code --------------------

def check_revenue_target(table: dict, query: str) -> tuple:
    """
    This function checks if the total revenue for last week met the target revenue.
    It uses the pre-trained model 'google/tapas-small-finetuned-wtq' from the transformers library.

    Args:
        table (dict): A dictionary representing the table containing the daily revenue for the week.
        query (str): A string representing the query asking whether the target revenue has been achieved.

    Returns:
        tuple: A tuple containing the predicted answer coordinates and aggregation indices.
    """
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-small-finetuned-wtq')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-small-finetuned-wtq')
    inputs = tokenizer(table=table, queries=query, return_tensors='pt')
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())
    return predicted_answer_coordinates, predicted_aggregation_indices

# test_function_code --------------------

def test_check_revenue_target():
    """
    This function tests the check_revenue_target function.
    It uses a sample table and query for testing.
    """
    table = {"Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
             "Revenue": [2000, 2500, 3000, 3500, 4000, 4500, 5000]}
    query = "Did the total revenue meet the target revenue of 24000?"
    predicted_answer_coordinates, predicted_aggregation_indices = check_revenue_target(table, query)
    assert isinstance(predicted_answer_coordinates, list)
    assert isinstance(predicted_aggregation_indices, list)

# call_test_function_code --------------------

test_check_revenue_target()