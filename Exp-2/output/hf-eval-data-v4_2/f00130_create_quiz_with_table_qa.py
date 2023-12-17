# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def create_quiz_with_table_qa(table_data, questions_list):
    """
    Generates answers for a quiz based on questions and tabular data.
    
    Args:
        table_data (dict): The tabular data to be used for answering questions.
        questions_list (list): The list of questions to generate answers for.

    Returns:
        list: A list containing the answers to the questions.

    Raises:
        ValueError: If the table_data or questions_list is not provided.
    """
    # Validating inputs
    if not table_data or not isinstance(table_data, dict):
        raise ValueError('table_data must be a non-empty dictionary.')
    if not questions_list or not isinstance(questions_list, list):
        raise ValueError('questions_list must be a non-empty list.')
    # Initialize the TAPAS for table question answering
    tapas_pipeline = pipeline('table-question-answering', model='Meena/table-question-answering-tapas')
    # Using the pipeline to generate answers for the provided questions
    answers = tapas_pipeline(questions_list, table_data)
    return answers

# test_function_code --------------------

def test_create_quiz_with_table_qa():
    print("Testing started.")
    # Prepare mock data for testing
    table_data = {
        'headers': ['Country', 'Capital'],
        'data': [['Germany', 'Berlin'], ['France', 'Paris'], ['Italy', 'Rome']]
    }
    questions_list = ['What is the capital of Germany?', 'Which country's capital is Rome?']

    # Testing case 1
    print("Testing case [1/2] started.")
    answer_1 = create_quiz_with_table_qa(table_data, questions_list)
    assert answer_1 == ['Berlin', 'Italy'], f"Test case [1/2] failed: {answer_1}"

    # Testing case 2 with empty inputs
    print("Testing case [2/2] started.")
    try:
        create_quiz_with_table_qa({}, [])
        assert False, "Test case [2/2] failed: No ValueError for empty inputs."
    except ValueError as e:
        assert str(e) == "table_data must be a non-empty dictionary.", f"Test case [2/2] failed: {str(e)}"

    print("Testing finished.")

# call_test_function_line --------------------

test_create_quiz_with_table_qa()