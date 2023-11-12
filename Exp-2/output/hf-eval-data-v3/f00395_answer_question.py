# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# function_code --------------------

def answer_question(question: str, context: str) -> str:
    """
    This function uses a pre-trained model from Hugging Face Transformers to answer a question based on a given context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the question is asked.

    Returns:
        str: The answer to the question.
    """
    model = AutoModelForQuestionAnswering.from_pretrained('deepset/deberta-v3-large-squad2')
    tokenizer = AutoTokenizer.from_pretrained('deepset/deberta-v3-large-squad2')

    inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)
    output = model(**inputs)
    answer_start = output.start_logits.argmax().item()
    answer_end = output.end_logits.argmax().item()

    ans = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end+1]))
    return ans

# test_function_code --------------------

def test_answer_question():
    """
    This function tests the answer_question function with some test cases.
    """
    question1 = 'What is the capital of France?'
    context1 = 'Paris is the capital of France.'
    assert answer_question(question1, context1) == 'Paris'

    question2 = 'Who won the world cup in 2018?'
    context2 = 'The 2018 FIFA World Cup was won by France.'
    assert answer_question(question2, context2) == 'France'

    question3 = 'Who is the CEO of Tesla?'
    context3 = 'Elon Musk is the CEO of Tesla.'
    assert answer_question(question3, context3) == 'Elon Musk'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_answer_question()