# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# function_code --------------------

def answer_question(question, context):
    """
    Answers a question based on the given context using a pre-trained transformer model.

    Args:
        question (str): The question to be answered.
        context (str): The text containing the answer to the question.

    Returns:
        str: The answer extracted from the context.

    Raises:
        ValueError: If either question or context is empty.
    """
    if not question or not context:
        raise ValueError('The question or context should not be empty.')
    model = AutoModelForQuestionAnswering.from_pretrained('ahotrod/electra_large_discriminator_squad2_512')
    tokenizer = AutoTokenizer.from_pretrained('ahotrod/electra_large_discriminator_squad2_512')

    inputs = tokenizer(question, context, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item() + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return answer

# test_function_code --------------------

def test_answer_question():
    print('Testing started.')

    # Test case 1: Standard question
    print('Testing case [1/3] started.')
    question = 'What is the capital of France?'
    context = 'France is a country in Europe. Its capital is Paris.'
    expected_answer = 'Paris'
    assert answer_question(question, context) == expected_answer, f"Test case [1/3] failed: expected {{expected_answer}}, got {{answer_question(question, context)}}"

    # Test case 2: Question with no answer in context
    print('Testing case [2/3] started.')
    question = 'What is the largest planet in the Solar System?'
    context = 'Jupiter is the largest planet in our Solar System.'
    expected_answer = 'Jupiter'
    assert answer_question(question, context) == expected_answer, f"Test case [2/3] failed: expected {{expected_answer}}, got {{answer_question(question, context)}}"

    # Test case 3: Empty question
    print('Testing case [3/3] started.')
    try:
        answer_question('', context)
        assert False, 'Test case [3/3] failed: ValueError was not raised for empty question.'
    except ValueError:
        pass

    print('Testing finished.')


# call_test_function_line --------------------

test_answer_question()