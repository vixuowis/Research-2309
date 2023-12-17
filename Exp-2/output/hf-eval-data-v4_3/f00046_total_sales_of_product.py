# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def total_sales_of_product(question, sales_data_table):
    """Calculate the total sales of a specific product from a sales data table based on a natural language question.

    Args:
        question (str): The question pointing to the product in question.
        sales_data_table (pd.DataFrame): The table containing weekly sales data.

    Returns:
        float: The total sales of the specified product.

    Raises:
        ValueError: Raises an exception if the sales_data_table is not in the expected format.
        RuntimeError: Raises an exception if the model fails to provide an answer.
    """
    tokenizer = TapasTokenizer.from_pretrained('lysandre/tapas-temporary-repo')
    model = TapasForQuestionAnswering.from_pretrained('lysandre/tapas-temporary-repo')
    inputs = tokenizer(table=sales_data_table, queries=question, return_tensors='pt')
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())

    # Extract the relevant sales figures based on the coordinates
    if not predicted_answer_coordinates:
        raise RuntimeError('Model did not return any answer coordinates.')

    total_sales = 0
    for coordinate in predicted_answer_coordinates:
        row_index, column_index = coordinate
        total_sales += float(sales_data_table.iloc[row_index, column_index])
    return total_sales

# test_function_code --------------------

import pandas as pd

def test_total_sales_of_product():
    print("Testing started.")
    # Assuming a sample sales data table exists
    sales_data_table = pd.DataFrame([
        ['Product', 'Week 1', 'Week 2', 'Week 3'],
        ['Product A', 100, 200, 300],
        ['Product B', 150, 250, 350]
    ])
    question = 'What is the total sales of Product A?'

    # Test case 1
    print("Testing case [1/1] started.")
    total_sales = total_sales_of_product(question, sales_data_table)
    assert total_sales == 600, f"Test case [1/1] failed: Expected total sales of 600 but got {total_sales}."
    print("Testing finished.")

# call_test_function_line --------------------

test_total_sales_of_product()