# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_info_from_french_doc(context: str, question: str) -> str:
    '''
    Extracts specific information from a French business document using a multilingual question-answering model.

    Args:
        context (str): The French text document from which to extract information.
        question (str): The specific question in French to answer based on the context.

    Returns:
        str: The answer to the question based on the context.
    '''

    # load model
    french_question_answering = pipeline(task="question-answering", model="mrm8488/bert-small-finetuned-squadv2")
    
    # predict answers to question from document
    answer = french_question_answering({'context': context, 'question': question})['answer']

    return answer

# function_code --------------------

# test_function_code --------------------

def test_extract_info_from_french_doc():
    '''
    Tests the function extract_info_from_french_doc.
    '''
    context = 'Manuel Romero travaille dur dans le dépôt hugginface/transformers récemment.'
    question = 'Qui a travaillé dur pour hugginface/transformers récemment?'
    assert isinstance(extract_info_from_french_doc(context, question), str)
    context = 'La tour Eiffel est située à Paris.'
    question = 'Où se trouve la tour Eiffel?'
    assert isinstance(extract_info_from_french_doc(context, question), str)
    context = 'Le président actuel de la France est Emmanuel Macron.'
    question = 'Qui est le président actuel de la France?'
    assert isinstance(extract_info_from_french_doc(context, question), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_extract_info_from_french_doc()