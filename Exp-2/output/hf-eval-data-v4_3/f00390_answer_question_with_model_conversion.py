# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline, AutoModel, AutoTokenizer

# function_code --------------------

def answer_question_with_model_conversion(question, context, model_name, tokenizer_name):
    """
    Answers a provided question using a specified model and tokenizer.

    Args:
        question (str): The question to be answered.
        context (str): The context containing the answer to the question.
        model_name (str): The name of the pre-trained model to use.
        tokenizer_name (str): The name of the tokenizer to use.

    Returns:
        str: The answer to the question extracted from the context.

    Raises:
        ValueError: If any of the arguments are None or empty.
    """
    if not question or not context or not model_name or not tokenizer_name:
        raise ValueError('All arguments must be provided and non-empty.')
    nlp = pipeline('question-answering', model=AutoModel.from_pretrained(model_name), tokenizer=AutoTokenizer.from_pretrained(tokenizer_name))
    answer = nlp({'question': question, 'context': context})['answer']
    return answer

# test_function_code --------------------

def test_answer_question_with_model_conversion():
    print('Testing started.')
    # Define the inputs for the test cases
    question = 'Why is model conversion important?'
    context = 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    model_name = 'deepset/bert-large-uncased-whole-word-masking-squad2'
    tokenizer_name = 'deepset/bert-large-uncased-whole-word-masking-squad2'

    # Test case 1 - Check for correct working
    print('Testing case [1/2] started.')
    expected_answer = 'gives freedom to the user and allows people to easily switch between different frameworks'
    actual_answer = answer_question_with_model_conversion(question, context, model_name, tokenizer_name)
    assert actual_answer == expected_answer, f'Test case [1/2] failed: Expected {expected_answer}, got {actual_answer}'

    # Test case 2 - Check for error handling
    print('Testing case [2/2] started.')
    try:
        answer_question_with_model_conversion('', '', '', '')
        assert False, 'Test case [2/2] failed: ValueError not raised for empty arguments'
    except ValueError as e:
        assert str(e) == 'All arguments must be provided and non-empty.', f'Test case [2/2] failed: Incorrect error message {str(e)}'
    print('Testing finished.')

# call_test_function_line --------------------

test_answer_question_with_model_conversion()