# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# function_code --------------------

def get_answer(question: str, context: str) -> str:
    """
    This function takes a question and a context as input and returns the answer to the question based on the context.
    It uses the pre-trained ELECTRA_large_discriminator language model fine-tuned on the SQuAD2.0 dataset.

    Args:
        question (str): The question to be answered.
        context (str): The context in which to find the answer.

    Returns:
        str: The answer to the question based on the context.
    """
    model = AutoModelForQuestionAnswering.from_pretrained('ahotrod/electra_large_discriminator_squad2_512')
    tokenizer = AutoTokenizer.from_pretrained('ahotrod/electra_large_discriminator_squad2_512')
    inputs = tokenizer(question, context, return_tensors='pt')
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item() + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer

# test_function_code --------------------

def test_get_answer():
    """
    This function tests the get_answer function.
    """
    question1 = 'What is the capital of France?'
    context1 = 'France is a country in Europe. Its capital is Paris.'
    expected_answer1 = 'Paris'
    assert get_answer(question1, context1) == expected_answer1

    question2 = 'Who is the president of the United States?'
    context2 = 'The president of the United States is Joe Biden.'
    expected_answer2 = 'Joe Biden'
    assert get_answer(question2, context2) == expected_answer2

    question3 = 'What is the largest planet in the solar system?'
    context3 = 'The largest planet in the solar system is Jupiter.'
    expected_answer3 = 'Jupiter'
    assert get_answer(question3, context3) == expected_answer3

    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_answer()