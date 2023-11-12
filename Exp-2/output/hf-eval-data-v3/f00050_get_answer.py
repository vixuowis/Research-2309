# function_import --------------------

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# function_code --------------------

def get_answer(question: str, context: str) -> dict:
    """
    This function uses the tinyroberta model from deepset for Question and Answer.

    Args:
        question (str): The question to be answered.
        context (str): The context in which the answer is to be found.

    Returns:
        dict: A dictionary containing the answer, score, start and end index of the answer.
    """
    model_name = 'deepset/tiny-roberta-squad2'
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    QA_input = {
      'question': question,
      'context': context
    }

    return nlp(QA_input)

# test_function_code --------------------

def test_get_answer():
    question = 'Why is model conversion important?'
    context = 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    result = get_answer(question, context)
    assert 'answer' in result
    assert 'score' in result
    assert 'start' in result
    assert 'end' in result
    assert isinstance(result['answer'], str)
    assert isinstance(result['score'], float)
    assert isinstance(result['start'], int)
    assert isinstance(result['end'], int)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_get_answer()