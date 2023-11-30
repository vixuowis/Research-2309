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

    model_name = 'distilbert-base-cased-distilled-squad'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    inputs = tokenizer(question, context, return_tensors='pt')
    
    answer_start_scores, answer_end_scores = model(**inputs, return_dict=False)
    answer_start = int(answer_start_scores.argmax())
    answer_end = int(answer_end_scores.argmax()) + 1
    
    output = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return output


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