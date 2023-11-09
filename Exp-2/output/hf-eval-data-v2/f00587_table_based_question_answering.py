# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def table_based_question_answering(table: pd.DataFrame, query: str) -> str:
    """
    This function uses the TAPEX model from Hugging Face Transformers to answer queries based on the input table.

    Args:
        table (pd.DataFrame): The input table in the form of a pandas DataFrame.
        query (str): The query for which the answer needs to be found from the table.

    Returns:
        str: The answer to the query based on the input table.
    """
    tokenizer = TapexTokenizer.from_pretrained('microsoft/tapex-base-finetuned-wtq')
    model = BartForConditionalGeneration.from_pretrained('microsoft/tapex-base-finetuned-wtq')
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer[0]

# test_function_code --------------------

def test_table_based_question_answering():
    """
    This function tests the table_based_question_answering function.
    It uses a sample table and a query, and asserts the output to the expected answer.
    """
    data = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['athens', 'paris', 'st. louis', 'athens', 'beijing', 'london']
    }
    table = pd.DataFrame.from_dict(data)
    query = 'In which year did beijing host the Olympic Games?'
    expected_answer = '2008'
    assert table_based_question_answering(table, query) == expected_answer

# call_test_function_code --------------------

test_table_based_question_answering()