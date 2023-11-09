# function_import --------------------

from transformers import pipeline, RobertaForQuestionAnswering, RobertaTokenizer

# function_code --------------------

def get_covid_answer(question: str, context: str) -> str:
    """
    This function uses a pre-trained Roberta model to answer questions about COVID-19.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the question is to be answered.

    Returns:
        str: The answer to the question.
    """
    qa_pipeline = pipeline(
        'question-answering', 
        model=RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2-covid'), 
        tokenizer=RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2-covid')
    )
    answer = qa_pipeline({'question': question, 'context': context})
    return answer['answer']

# test_function_code --------------------

def test_get_covid_answer():
    """
    This function tests the get_covid_answer function.
    """
    question = 'What are the common symptoms of COVID-19?'
    context = 'COVID-19 is a respiratory disease with common symptoms such as cough, fever, and difficulty breathing.'
    answer = get_covid_answer(question, context)
    assert isinstance(answer, str), 'The answer should be a string.'
    assert answer.lower() in context.lower(), 'The answer should be in the context.'

# call_test_function_code --------------------

test_get_covid_answer()