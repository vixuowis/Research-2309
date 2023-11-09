# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

# function_code --------------------

def get_answer_from_table(table: pd.DataFrame, query: str) -> str:
    """
    This function uses a pretrained model to answer a question based on the information in a table.

    Args:
        table (pd.DataFrame): The table containing the information.
        query (str): The question to be answered.

    Returns:
        str: The answer to the question.
    """
    tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large-finetuned-wtq')
    model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large-finetuned-wtq')
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# test_function_code --------------------

def test_get_answer_from_table():
    """
    This function tests the get_answer_from_table function.
    """
    data = {
        'Animal': ['Tiger', 'Lion', 'Giraffe', 'Elephant'],
        'Habitat': ['Forest', 'Grassland', 'Savanna', 'Savanna'],
        'Average Lifespan': [10, 12, 25, 50],
    }
    table = pd.DataFrame.from_dict(data)
    query = 'What is the average lifespan of a giraffe?'
    answer = get_answer_from_table(table, query)
    assert isinstance(answer, str), 'The function should return a string.'
    assert answer.isdigit(), 'The answer should be a number.'

# call_test_function_code --------------------

test_get_answer_from_table()