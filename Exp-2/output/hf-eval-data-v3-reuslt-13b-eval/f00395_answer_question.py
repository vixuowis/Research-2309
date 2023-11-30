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

    model_name = "valhalla/t5-small-qa-qg-hl"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    encoding = tokenizer(question, context, return_tensors="pt")
    output = model(**encoding)
    
    answers = [tokenizer.decode(ans) for ans in output.start_logits[0].topk(1).indices]
    
    # We can only expect a single answer here, so just grab the first item from the list.
    return answers[0]


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