# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

# function_code --------------------

def get_hot_chocolate_shops_and_prices(table_data, query):
    """
    This function identifies which shops sell hot chocolate and their prices from a given table using the TAPAS model.

    Args:
        table_data (list of list of str): The table containing shop information and drink prices.
        query (str): The query to be answered by the TAPAS model.

    Returns:
        dict: A dictionary with shop names as keys and hot chocolate prices as values.

    Raises:
        ValueError: If the table_data is empty or not properly formatted.

    """
    if not table_data or not all(len(row) == 3 for row in table_data):
        raise ValueError('Table data is missing or improperly formatted.')

    model = TapasForQuestionAnswering.from_pretrained('google/tapas-mini-finetuned-sqa')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-mini-finetuned-sqa')

    dataframe = pd.DataFrame(table_data[1:], columns=table_data[0])
    inputs = tokenizer(table=dataframe, queries=[query], padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)

    answered_shops = [table_data[row_idx][0] for row_idx in outputs['answer_coordinates'][0][:, 0]]
    hot_chocolate_prices = [table_data[row_idx][2] for row_idx in outputs['answer_coordinates'][0][:, 0]]

    answer = {shop: price for shop, price in zip(answered_shops, hot_chocolate_prices)}
    return answer

# test_function_code --------------------

def test_get_hot_chocolate_shops_and_prices():
    print("Testing started.")
    table_data = [["Shop", "Drink", "Price"], ["Cafe A", "Coffee", "3.00"], ["Cafe B", "Tea", "2.50"], ["Cafe C", "Hot Chocolate", "4.50"], ["Cafe D", "Hot Chocolate", "3.75"]]
    query = "Which shops sell hot chocolate and what are their prices?"

    # Test case 1: Function returns correct output
    print("Testing case [1/1] started.")
    result = get_hot_chocolate_shops_and_prices(table_data, query)
    expected = {'Cafe C': '4.50', 'Cafe D': '3.75'}
    assert result == expected, f"Test case [1/1] failed: Expected {expected}, got {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_get_hot_chocolate_shops_and_prices()