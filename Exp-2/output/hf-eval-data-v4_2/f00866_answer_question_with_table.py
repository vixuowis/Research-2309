# requirements_file --------------------

!pip install -U pandas transformers

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def answer_question_with_table(table_data, query):
    """
    Answers the question based on the given table data using the TAPAS model.

    Args:
        table_data (pd.DataFrame): The table containing the relevant data.
        query (str): The question to be answered based on the table data.

    Returns:
        str: The answer to the question derived from the table.

    Raises:
        ValueError: If the table_data is not a pandas DataFrame or query is not a string.
    """
    if not isinstance(table_data, pd.DataFrame) or not isinstance(query, str):
        raise ValueError('The table_data must be a pandas DataFrame and the query must be a string.')

    model_name = 'google/tapas-base-finetuned-wtq'
    model = TapasForQuestionAnswering.from_pretrained(model_name)
    tokenizer = TapasTokenizer.from_pretrained(model_name)

    inputs = tokenizer(table_data, query, return_tensors='pt')
    outputs = model(**inputs)

    # Get the most probable answer tokens
    predicted_answer_ids = inputs['input_ids'][0, outputs.logits.argmax(-1).item()].numpy()
    answer = tokenizer.decode(predicted_answer_ids, skip_special_tokens=True)

    return answer

# test_function_code --------------------

def test_answer_question_with_table():
    import pandas as pd
    print("Testing started.")
    # Sample table data
    data = {
        'Region': ['North', 'South', 'East', 'West'],
        'Salesperson': ['Alice', 'Bob', 'Charlie', 'David'],
        'Sales': [560, 340, 610, 480]
    }
    table_data = pd.DataFrame(data)

    # Testing case 1: Valid input
    print("Testing case [1/3] started.")
    query1 = 'Who had the highest sales?'
    answer1 = answer_question_with_table(table_data, query1)
    assert answer1 == 'Charlie', f"Test case [1/3] failed: Expected 'Charlie', but got {answer1}"

    # Testing case 2: Valid input different question
    print("Testing case [2/3] started.")
    query2 = 'What was the total sales in East region?'
    answer2 = answer_question_with_table(table_data, query2)
    assert answer2 == '610', f"Test case [2/3] failed: Expected '610', but got {answer2}"

    # Testing case 3: Invalid table data type
    print("Testing case [3/3] started.")
    with pytest.raises(ValueError):
        answer_question_with_table('not a dataframe', query1)

    print("Testing finished.")

# call_test_function_line --------------------

test_answer_question_with_table()