# function_import --------------------

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def table_question_answering(csv_file, query):
    """
    This function takes a CSV file and a query as input, and returns the answer to the query based on the table in the CSV file.
    
    Args:
        csv_file (str): The path to the CSV file.
        query (str): The query to be answered.
    
    Returns:
        str: The answer to the query.
    """
    tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large-1024shot')
    model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large-1024shot')
    table = pd.read_csv(csv_file)
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer

# test_function_code --------------------

def test_table_question_answering():
    """
    This function tests the table_question_answering function.
    """
    csv_file = 'test.csv'
    query = 'test_query'
    expected_answer = 'test_answer'
    assert table_question_answering(csv_file, query) == expected_answer

# call_test_function_code --------------------

test_table_question_answering()