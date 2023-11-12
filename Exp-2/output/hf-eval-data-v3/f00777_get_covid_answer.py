# function_import --------------------

from transformers import pipeline, RobertaForQuestionAnswering, RobertaTokenizer

# function_code --------------------

def get_covid_answer(question: str, context: str) -> str:
    """
    This function uses a pre-trained model to answer questions related to COVID-19.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the answer is to be found.

    Returns:
        str: The answer to the question.
    """
    nlp = pipeline('question-answering', model=RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2-covid'), tokenizer=RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2-covid'))
    QA_input = {'question': question, 'context': context}
    answer = nlp(QA_input)
    return answer['answer']

# test_function_code --------------------

def test_get_covid_answer():
    """
    This function tests the get_covid_answer function.
    """
    question = 'What are the symptoms of COVID-19?'
    context = 'The most common symptoms of COVID-19 include fever, dry cough, and shortness of breath. Some patients may also experience fatigue, headache, and muscle pain.'
    assert get_covid_answer(question, context) == 'The most common symptoms of COVID-19 include fever, dry cough, and shortness of breath. Some patients may also experience fatigue, headache, and muscle pain.'
    question = 'What is COVID-19?'
    context = 'COVID-19 is a novel coronavirus that was first identified in Wuhan, China in December 2019.'
    assert get_covid_answer(question, context) == 'COVID-19 is a novel coronavirus that was first identified in Wuhan, China in December 2019.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_covid_answer()