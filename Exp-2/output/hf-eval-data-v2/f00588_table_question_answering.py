# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def table_question_answering(table, question):
    """
    This function receives a table and a question in natural language, and returns an answer to the question based on the inputted table.
    
    Args:
    table (str): The table to be queried.
    question (str): The question in natural language.
    
    Returns:
    predicted_answer_coordinates (list): The coordinates of the predicted answer in the table.
    predicted_aggregation_indices (list): The aggregation indices of the predicted answer.
    """
    tokenizer = TapasTokenizer.from_pretrained('lysandre/tapas-temporary-repo')
    model = TapasForQuestionAnswering.from_pretrained('lysandre/tapas-temporary-repo')
    inputs = tokenizer(table=table, queries=question, return_tensors='pt')
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())
    return predicted_answer_coordinates, predicted_aggregation_indices

# test_function_code --------------------

def test_table_question_answering():
    """
    This function tests the table_question_answering function with a sample table and question.
    """
    table = """\nCategory    | Sub-Category    | Revenue\n---         | ---             | ---\nFurniture   | Bookcases       | 114879\nFurniture   | Chairs          | 328449\nTechnology  | Phones          | 330007\n"""
    question = "What is the Revenue of Phones?"
    predicted_answer_coordinates, predicted_aggregation_indices = table_question_answering(table, question)
    assert predicted_answer_coordinates == [(2, 2)], "The predicted answer coordinates are incorrect."
    assert predicted_aggregation_indices == [0], "The predicted aggregation indices are incorrect."

# call_test_function_code --------------------

test_table_question_answering()