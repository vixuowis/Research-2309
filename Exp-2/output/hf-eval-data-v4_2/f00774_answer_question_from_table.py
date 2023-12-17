# requirements_file --------------------

!pip install -U transformers pandas

# function_import --------------------

from transformers import TapasForQuestionAnswering, TapasTokenizer

# function_code --------------------

def answer_question_from_table(question, table_data):
    """
    Answers a question based on the provided table data using the TAPAS model.

    Args:
        question (str): The question to be answered.
        table_data (pd.DataFrame): The structured table data in the form of a pandas DataFrame.

    Returns:
        list: A list containing the answer to the question.

    Raises:
        ValueError: If the question is empty or table_data is none.
    """
    # Validate inputs
    if not question or table_data is None:
        raise ValueError('The question or table_data cannot be empty or None.')

    # Load model and tokenizer
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wikisql-supervised')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wikisql-supervised')

    # Tokenize the question and table data
    inputs = tokenizer(table_data, queries=question, return_tensors='pt', padding='max_length', truncation=True)

    # Perform inference
    outputs = model(**inputs)

    # Decode the predicted answer
    predicted_answer = tokenizer.decode(outputs.logits.argmax(dim=-1))

    return [predicted_answer]

# test_function_code --------------------

def test_answer_question_from_table():
    import pandas as pd
    print("Testing started.")

    # Create a sample table
    sample_data = {'column1': ['item1', 'item2'], 'column2': [1, 2]}
    table_data = pd.DataFrame(sample_data)

    # Testing case 1: Valid question and table_data
    print("Testing case [1/2] started.")
    result = answer_question_from_table("What is the value in column2 for item1?", table_data)
    assert result == ['1'], f"Test case [1/2] failed: Expected ['1'], got {result}"

    # Testing case 2: Empty question
    print("Testing case [2/2] started.")
    try:
        answer_question_from_table("", table_data)
        assert False, "Test case [2/2] failed: Expected ValueError for empty question."
    except ValueError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_answer_question_from_table()