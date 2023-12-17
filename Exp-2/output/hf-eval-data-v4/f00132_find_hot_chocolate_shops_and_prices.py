# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

# function_code --------------------

def find_hot_chocolate_shops_and_prices(table_data, query):
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-mini-finetuned-sqa')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-mini-finetuned-sqa')

    dataframe = pd.DataFrame(table_data[1:], columns=table_data[0])
    inputs = tokenizer(table=dataframe, queries=[query], padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)

    answered_shops = [dataframe.iloc[row_idx]['Shop'] for row_idx in outputs['answer_coordinates'][0][:, 0]]
    hot_chocolate_prices = [dataframe.iloc[row_idx]['Price'] for row_idx in outputs['answer_coordinates'][0][:, 0]]

    return dict(zip(answered_shops, hot_chocolate_prices))

# test_function_code --------------------

def test_find_hot_chocolate_shops_and_prices():
    table_data = [['Shop', 'Drink', 'Price'], ['Cafe A', 'Coffee', '3.00'], ['Cafe B', 'Tea', '2.50'], ['Cafe C', 'Hot Chocolate', '4.50'], ['Cafe D', 'Hot Chocolate', '3.75']]
    query = 'Which shops sell hot chocolate and what are their prices?'
    expected_result = {'Cafe C': '4.50', 'Cafe D': '3.75'}

    result = find_hot_chocolate_shops_and_prices(table_data, query)
    assert result == expected_result, f'Test failed: expected {expected_result}, got {result}'

    print('Test succeeded')

test_find_hot_chocolate_shops_and_prices()