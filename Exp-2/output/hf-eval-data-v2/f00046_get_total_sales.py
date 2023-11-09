# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def get_total_sales(sales_data_table, question):
    """
    This function uses the TAPAS model from Hugging Face Transformers to answer questions related to a table.
    The model is pretrained on a large corpus of English data from Wikipedia in a self-supervised fashion.
    It is fine-tuned on Sequential Question Answering (SQA).

    Args:
    sales_data_table (pd.DataFrame): The sales data table.
    question (str): The question to be answered.

    Returns:
    Tuple: The predicted answer coordinates and aggregation indices.
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
    This function tests the get_total_sales function.
    It uses a sample sales data table and a sample question.
    """
    sales_data_table = pd.DataFrame({'Product': ['A', 'B', 'C'], 'Sales': [100, 200, 300], 'Week': ['1', '1', '1']})
    question = 'What is the total sales of Product A?'
    predicted_answer_coordinates, predicted_aggregation_indices = get_total_sales(sales_data_table, question)
    assert predicted_answer_coordinates is not None
    assert predicted_aggregation_indices is not None

# call_test_function_code --------------------

test_get_total_sales()