# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def get_sales_extremes(sales_data_table):
    """
    This function uses the TapasForQuestionAnswering model from Hugging Face Transformers to answer the question
    'What are the highest and lowest sales numbers?' based on the provided sales data table.

    Args:
        sales_data_table (pd.DataFrame): The sales data table.

    Returns:
        tuple: The highest and lowest sales numbers.
    """
    tokenizer = TapasTokenizer.from_pretrained('lysandre/tapas-temporary-repo')
    model = TapasForQuestionAnswering.from_pretrained('lysandre/tapas-temporary-repo')
    inputs = tokenizer(table=sales_data_table, queries="What are the highest and lowest sales numbers?", return_tensors='pt')
    outputs = model(**inputs)
    predicted_answer_coordinates, _ = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())
    highest_sales, lowest_sales = extract_aggregated_sales(predicted_answer_coordinates)
    return highest_sales, lowest_sales

# test_function_code --------------------

def test_get_sales_extremes():
    """
    This function tests the get_sales_extremes function by using a sample sales data table.
    """
    sales_data_table = pd.DataFrame({'Product': ['A', 'B', 'C', 'D', 'E'], 'Sales': [100, 200, 300, 400, 500]})
    highest_sales, lowest_sales = get_sales_extremes(sales_data_table)
    assert highest_sales == 500 and lowest_sales == 100, 'Test failed!'
    print('Test passed!')

# call_test_function_code --------------------

test_get_sales_extremes()