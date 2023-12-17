# requirements_file --------------------

import subprocess

requirements = ["transformers", "pandas"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def aggregate_sales_data(sales_data_table, question):
    """
    Extracts the highest and lowest sales numbers from a given sales data table using TAPAS.

    Args:
        sales_data_table (pd.DataFrame): The table containing sales data.
        question (str): Query to ask the model for extracting the sales data.

    Returns:
        dict: A dictionary with 'highest_sales' and 'lowest_sales'.

    Raises:
        ValueError: If the sales_data_table is empty or question is not provided.
    """
    if sales_data_table.empty:
        raise ValueError('The sales_data_table is empty.')
    if not question:
        raise ValueError('The question is not provided.')
    
    tokenizer = TapasTokenizer.from_pretrained('lysandre/tapas-temporary-repo')
    model = TapasForQuestionAnswering.from_pretrained('lysandre/tapas-temporary-repo')
    inputs = tokenizer(table=sales_data_table, queries=question, return_tensors='pt')
    outputs = model(**inputs)
    
    predicted_answer_coordinates, _ = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())
    highest_sales, lowest_sales = extract_aggregated_sales(predicted_answer_coordinates)

    return {'highest_sales': highest_sales, 'lowest_sales': lowest_sales}

# test_function_code --------------------

def test_aggregate_sales_data():
    print('Testing started.')
    # Load a sample sales_data_table and define a question
    sales_data_table = pd.DataFrame(...)  # Define the structure with mock data
    question = 'What are the highest and lowest sales numbers?'

    # Testing case 1
    print('Testing case [1/2] started.')
    result = aggregate_sales_data(sales_data_table, question)
    assert 'highest_sales' in result and 'lowest_sales' in result, f'Test case [1/2] failed: the function did not return the expected keys.'

    # Testing case 2: No sales data
    print('Testing case [2/2] started.')
    try:
        aggregate_sales_data(pd.DataFrame(), question)
        assert False, 'Test case [2/2] failed: expected a ValueError for empty sales data table.'
    except ValueError:
        pass

    print('Testing finished.')

# call_test_function_line --------------------

test_aggregate_sales_data()