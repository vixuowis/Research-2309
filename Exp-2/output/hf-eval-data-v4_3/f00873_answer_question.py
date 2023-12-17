# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_question(question: str, context: str) -> str:
    """
    Answers a question based on the provided context using a pretrained Korean question-answering model.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the answer will be searched for.

    Returns:
        str: The answer extracted from the context.

    Raises:
        ValueError: If the question or context is not provided.
    """
    if not question or not context:
        raise ValueError('Question and context must be provided')

    # Load the pretrained Korean QA model
    korean_qa = pipeline('question-answering', model='monologg/koelectra-small-v2-distilled-korquad-384')

    # Extract the answer
    answer = korean_qa(question=question, context=context)

    return answer['answer']

# test_function_code --------------------

def test_answer_question():
    print("Testing started.")
    question = '대한민국의 수도는 어디인가요?'
    context = '대한민국의 수도는 서울이다. 서울은 한국의 가장 큰 도시이며, 정치, 경제, 문화의 중심지이다.'

    # Test case 1: Standard question
    print("Testing case [1/1] started.")
    expected_answer = '서울'
    actual_answer = answer_question(question, context)
    assert expected_answer == actual_answer, f"Test case [1/1] failed: Expected {expected_answer}, got {actual_answer}"
    print("Testing finished.")

# call_test_function_line --------------------

test_answer_question()