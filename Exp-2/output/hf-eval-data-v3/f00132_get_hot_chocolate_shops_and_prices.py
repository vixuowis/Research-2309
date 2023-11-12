# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

# function_code --------------------

def get_hot_chocolate_shops_and_prices(table: list, queries: list) -> dict:
    """
    Get the shops that sell hot chocolate and their prices.

    Args:
        table (list): A list of lists representing the table of shops, drinks, and prices.
        queries (list): A list of queries to ask the model.

    Returns:
        dict: A dictionary where the keys are the shops and the values are the prices of hot chocolate.

    Raises:
        KeyError: If 'answer_coordinates' is not in the model's output.
    """
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-mini-finetuned-sqa')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-mini-finetuned-sqa')

    dataframe = pd.DataFrame(table[1:], columns=table[0])
    inputs = tokenizer(table=dataframe, queries=queries, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)

    answered_shops = [table[row_idx][0] for row_idx in outputs['answer_coordinates'][0][:, 0]]
    hot_chocolate_prices = [table[row_idx][2] for row_idx in outputs['answer_coordinates'][0][:, 0]]

    return {shop: price for shop, price in zip(answered_shops, hot_chocolate_prices)}

# test_function_code --------------------

def test_get_hot_chocolate_shops_and_prices():
    """Test the function get_hot_chocolate_shops_and_prices."""
    table = [["Shop", "Drink", "Price"], ["Cafe A", "Coffee", "3.00"], ["Cafe B", "Tea", "2.50"], ["Cafe C", "Hot Chocolate", "4.50"], ["Cafe D", "Hot Chocolate", "3.75"]]
    queries = ["Which shops sell hot chocolate and what are their prices?"]
    expected_output = {"Cafe C": "4.50", "Cafe D": "3.75"}
    assert get_hot_chocolate_shops_and_prices(table, queries) == expected_output

    table = [["Shop", "Drink", "Price"], ["Cafe A", "Coffee", "3.00"], ["Cafe B", "Tea", "2.50"], ["Cafe E", "Hot Chocolate", "5.00"]]
    queries = ["Which shops sell hot chocolate and what are their prices?"]
    expected_output = {"Cafe E": "5.00"}
    assert get_hot_chocolate_shops_and_prices(table, queries) == expected_output

    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_hot_chocolate_shops_and_prices()