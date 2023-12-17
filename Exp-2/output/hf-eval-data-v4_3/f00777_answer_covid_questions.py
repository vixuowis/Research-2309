# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline, RobertaForQuestionAnswering, RobertaTokenizer

# function_code --------------------

def answer_covid_questions(question, context):
    """
    Answers questions related to COVID-19 by using a pre-trained model.

    Args:
        question (str): The question regarding COVID-19.
        context (str): The context text related to the question.

    Returns:
        str: The answer to the question based on the context.

    Raises:
        ValueError: If either question or context is empty.
    """
    if not question or not context:
        raise ValueError("Both question and context must be provided.")

    nlp = pipeline('question-answering', model=RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2-covid'), tokenizer=RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2-covid'))
    QA_input = {'question': question, 'context': context}
    answer = nlp(QA_input)
    return answer['answer']

# test_function_code --------------------

def test_answer_covid_questions():
    print("Testing started.")

    # Test case 1: Standard context and question
    print("Testing case [1/2] started.")
    context1 = "COVID-19 is known to cause fever, cough, and difficulty breathing."
    question1 = "What are symptoms of COVID-19?"
    expected_answer1 = "fever, cough, and difficulty breathing"
    answer1 = answer_covid_questions(question1, context1)
    assert expected_answer1 in answer1, f"Test case [1/2] failed: expected {expected_answer1}, got {answer1}"

    # Test case 2: Empty question or context
    print("Testing case [2/2] started.")
    context2 = ""
    question2 = "Is COVID-19 contagious?"
    try:
        answer_covid_questions(question2, context2)
        assert False, "Test case [2/2] failed: ValueError expected"
    except ValueError as e:
        assert str(e) == "Both question and context must be provided."

    print("Testing finished.")

# call_test_function_line --------------------

test_answer_covid_questions()