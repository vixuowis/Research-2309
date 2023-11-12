# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

# function_code --------------------

def table_based_qa(table: pd.DataFrame, query: str) -> str:
    """
    Extracts key information from a table by asking a natural language question.

    Args:
        table (pd.DataFrame): The table from which to extract information.
        query (str): The natural language question to ask.

    Returns:
        str: The answer to the question.
    """
    tokenizer = AutoTokenizer.from_pretrained('neulab/omnitab-large-1024shot')
    model = AutoModelForSeq2SeqLM.from_pretrained('neulab/omnitab-large-1024shot')
    encoding = tokenizer(table=table, query=query, return_tensors='pt')
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer[0]

# test_function_code --------------------

def test_table_based_qa():
    """Tests the table_based_qa function."""
    data = {'year': [1896, 1900, 1904, 2004, 2008, 2012], 'city': ['Athens', 'Paris', 'St. Louis', 'Athens', 'Beijing', 'London']}
    table = pd.DataFrame.from_dict(data)
    assert table_based_qa(table, 'In which year did Beijing host the Olympic Games?') == '2008'
    assert table_based_qa(table, 'In which city did the 1896 Olympic Games take place?') == 'Athens'
    assert table_based_qa(table, 'In which year did London host the Olympic Games?') == '2012'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_table_based_qa()