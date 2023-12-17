# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def create_quiz_with_table_qa(table_data, questions_list):
    """
    This function generates answers for a list of questions based on the given table data.

    :param table_data: A dictionary representing the table with headers as keys and lists of cell data as values.
    :param questions_list: A list of questions to be answered based on the table data.
    :return: A dictionary with questions as keys and their corresponding answers.
    """
    tapas_pipeline = pipeline('table-question-answering', model='Meena/table-question-answering-tapas')
    return tapas_pipeline(questions_list, table_data)

# test_function_code --------------------

def test_create_quiz_with_table_qa():
    print("Testing started.")

    # Sample table data, replace with real data
    table_data = {
        'Column1': ['Row1', 'Row2', 'Row3'],
        'Column2': [123, 456, 789]
    }

    # List of sample questions, replace with real questions
    questions_list = [
        'What is in Column1 of Row2?',
        'What is the value in Column2 of Row3?'
    ]

    # Expected answers, replace with expected real answers after executing the model
    expected_answers = {
        'What is in Column1 of Row2?': 'Row2',
        'What is the value in Column2 of Row3?': 789
    }

    # Testing the function
    print("Testing the create_quiz_with_table_qa function.")
    answers = create_quiz_with_table_qa(table_data, questions_list)
    assert answers == expected_answers, f"Test failed: {answers} does not match {expected_answers}"

    print("Testing finished.")

# Run the test function
test_create_quiz_with_table_qa()