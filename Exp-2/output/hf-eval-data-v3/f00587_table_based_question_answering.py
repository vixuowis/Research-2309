# function_import --------------------

from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# function_code --------------------

def table_based_question_answering(table: pd.DataFrame, query: str) -> str:
    """
    This function takes a pandas DataFrame and a query as input, and returns the answer to the query based on the table.

    Args:
        table (pd.DataFrame): The input table in pandas DataFrame format.
        query (str): The query for which the answer needs to be found from the table.

    Returns:
        str: The answer to the query based on the table.
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
    This function tests the table_based_question_answering function by using some test cases.
    """
    data1 = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['athens', 'paris', 'st. louis', 'athens', 'beijing', 'london']
    }
    table1 = pd.DataFrame.from_dict(data1)
    query1 = 'In which year did beijing host the Olympic Games?'
    assert table_based_question_answering(table1, query1) == '2008'

    data2 = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['athens', 'paris', 'st. louis', 'athens', 'beijing', 'london']
    }
    table2 = pd.DataFrame.from_dict(data2)
    query2 = 'In which year did athens host the Olympic Games?'
    assert '1896' in table_based_question_answering(table2, query2)

    data3 = {
        'year': [1896, 1900, 1904, 2004, 2008, 2012],
        'city': ['athens', 'paris', 'st. louis', 'athens', 'beijing', 'london']
    }
    table3 = pd.DataFrame.from_dict(data3)
    query3 = 'In which year did london host the Olympic Games?'
    assert '2012' in table_based_question_answering(table3, query3)

    return 'All Tests Passed'

# call_test_function_code --------------------

test_table_based_question_answering()