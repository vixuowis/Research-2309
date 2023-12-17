# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_korean_table_question(table, question_korean):
    """
    Answers a question in Korean based on the provided table data using Hugging Face's transformers pipeline.

    Args:
        table (dict): A dictionary representing the table to query against.
        question_korean (str): The question in Korean to be answered based on the table data.

    Returns:
        dict: The answer provided by the table-question-answering model.

    Raises:
        ValueError: If the input table is not a dictionary or question_korean is not a string.
    """
    # Validate input parameters
    if not isinstance(table, dict):
        raise ValueError('Input table must be a dictionary.')
    if not isinstance(question_korean, str):
        raise ValueError('Question must be a string.')

    # Initialize the table QA pipeline
    table_qa = pipeline('table-question-answering', model='dsba-lab/koreapas-finetuned-korwikitq')

    # Query the model
    return table_qa(table=table, query=question_korean)

# test_function_code --------------------

def test_answer_korean_table_question():
    print("Testing started.")

    # Sample data for testing
    sample_table = {'데이터셋': ['데이터1', '데이터2', '데이터3'],
                    '정보': [10, 20, 30]}
    sample_question = '데이터셋 중에서 정보가 20인 것은 무엇인가요?'

    # Test case 1
    print("Testing case [1/1] started.")
    result = answer_korean_table_question(sample_table, sample_question)
    expected_answer = '데이터2'
    assert result['answer'] == expected_answer, f"Test case [1/1] failed: Expected {expected_answer}, got {result['answer']}"
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_korean_table_question()