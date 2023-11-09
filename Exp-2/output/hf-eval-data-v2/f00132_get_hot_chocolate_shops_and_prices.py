# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

# function_code --------------------

def get_hot_chocolate_shops_and_prices(table, queries):
    """
    This function uses the TapasForQuestionAnswering model from the transformers library to answer questions about a given table.
    The model is pretrained for sequential question answering.
    
    Args:
    table (list): A list of lists representing the table. The first list is the header, and the following lists are the rows.
    queries (list): A list of strings representing the questions to be asked.
    
    Returns:
    dict: A dictionary where the keys are the shops that sell hot chocolate and the values are their prices.
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
    """
    This function tests the get_hot_chocolate_shops_and_prices function.
    It uses a sample table and queries, and checks if the returned dictionary is correct.
    """
    table = [["Shop", "Drink", "Price"], ["Cafe A", "Coffee", "3.00"], ["Cafe B", "Tea", "2.50"], ["Cafe C", "Hot Chocolate", "4.50"], ["Cafe D", "Hot Chocolate", "3.75"]]
    queries = ["Which shops sell hot chocolate and what are their prices?"]
    expected_output = {"Cafe C": "4.50", "Cafe D": "3.75"}
    
    assert get_hot_chocolate_shops_and_prices(table, queries) == expected_output

# call_test_function_code --------------------

test_get_hot_chocolate_shops_and_prices()