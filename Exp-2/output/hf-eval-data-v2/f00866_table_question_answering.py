# function_import --------------------

from transformers import TapasForQuestionAnswering, TapasTokenizer

# function_code --------------------

def table_question_answering(table, query):
    """
    This function uses the TAPAS (Tabular Pretraining for Answering Questions) model from Hugging Face's transformers library
    to answer questions based on a given table.

    Args:
        table (pd.DataFrame): The table data in pandas DataFrame format.
        query (str): The question to be answered based on the table data.

    Returns:
        str: The answer to the question.
    """
    # Load the TAPAS model and tokenizer
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wtq')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')

    # Tokenize the table and query
    inputs = tokenizer(table=table, queries=query, padding='max_length', return_tensors='pt')

    # Get the model outputs
    outputs = model(**inputs)

    # Get the predicted answer
    predicted_answer = tokenizer.convert_ids_to_tokens(outputs.predicted_answer_ids)[0]

    return predicted_answer

# test_function_code --------------------

def test_table_question_answering():
    """
    This function tests the table_question_answering function by using a sample table and query.
    """
    # Define a sample table
    table = pd.DataFrame({'Salesperson': ['John Doe', 'Jane Doe'], 'Region': ['West', 'East'], 'Sales': [500, 600]})

    # Define a sample query
    query = 'Who had the highest sales?'

    # Get the answer from the table_question_answering function
    answer = table_question_answering(table, query)

    # Assert that the answer is correct
    assert answer == 'Jane Doe', f'Expected Jane Doe, but got {answer}'

# call_test_function_code --------------------

test_table_question_answering()