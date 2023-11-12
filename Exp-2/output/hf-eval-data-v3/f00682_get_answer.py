# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def get_answer(question: str, context: str) -> str:
    """
    This function takes a question and a context as input and returns the answer to the question based on the context.
    It uses the 'deepset/roberta-base-squad2' model from Hugging Face Transformers for question answering.

    Args:
        question (str): The question to be answered.
        context (str): The context in which to find the answer.

    Returns:
        str: The answer to the question.
    """
    model_name = 'deepset/roberta-base-squad2'
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
     'question': question,
     'context': context
    }
    res = nlp(QA_input)
    return res['answer']

# test_function_code --------------------

def test_get_answer():
    assert get_answer('What was the main cause of the war?', 'World War I was primarily caused by a complex web of factors including political, economic, and social issues. However, the assassination of Archduke Franz Ferdinand of Austria is often cited as the immediate trigger for the conflict.') == 'the assassination of Archduke Franz Ferdinand of Austria'
    assert get_answer('Who won the world series in 2020?', 'The Los Angeles Dodgers won the World Series in 2020.') == 'The Los Angeles Dodgers'
    assert get_answer('Who is the president of the United States?', 'As of 2021, the president of the United States is Joe Biden.') == 'Joe Biden'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_answer()