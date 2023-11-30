# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(context: str, question: str) -> str:
    """
    This function uses the Hugging Face Transformers pipeline for question answering.
    It uses the 'sultan/BioM-ELECTRA-Large-SQuAD2' model which is specialized in biomedical language.

    Args:
        context (str): The context in which the question is being asked.
        question (str): The question that needs to be answered.

    Returns:
        str: The answer to the question based on the provided context.
    """
    # load model
    qa_sultan = pipeline("question-answering", \
                         model="sultan/BioM-ELECTRA-Large-SQuAD2", \
                         tokenizer="sultan/BioM-ELECTRA-Large-SQuAD2")
    
    # get answer
    answer = qa_sultan({'context': context, 'question': question})['answer']
    
    return answer

# test_function_code --------------------

def test_get_answer():
    """
    This function tests the 'get_answer' function with some test cases.
    """
    assert get_answer('Paracetamol is a common pain reliever.', 'What is a common pain reliever?') == 'Paracetamol'
    assert get_answer('The heart is an organ that pumps blood.', 'What is the function of the heart?') == 'pumps blood'
    assert get_answer('The sun is a star.', 'What is the sun?') == 'a star'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_get_answer()