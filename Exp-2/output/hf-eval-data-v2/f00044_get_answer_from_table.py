# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer_from_table(question: str, table_data: dict) -> str:
    """
    This function uses the TAPAS small model fine-tuned on Sequential Question Answering (SQA) from the transformers library
    to answer questions based on tabular data.

    Args:
        question (str): The question to be answered based on the table data.
        table_data (dict): The table data in the form of a dictionary.

    Returns:
        str: The answer to the question based on the table data.
    """
    table_qa = pipeline('table-question-answering', model='google/tapas-small-finetuned-sqa')
    answer = table_qa(question=question, table=table_data)
    return answer

# test_function_code --------------------

def test_get_answer_from_table():
    """
    This function tests the get_answer_from_table function by using a sample question and table data.
    """
    question = 'What is the total revenue for product ID 12345?'
    table_data = {'Product ID': ['12345', '67890'], 'Revenue': [1000, 2000]}
    answer = get_answer_from_table(question, table_data)
    assert isinstance(answer, str), 'The function should return a string.'

# call_test_function_code --------------------

test_get_answer_from_table()