# function_import --------------------

from transformers import AutoModel, pipeline

# function_code --------------------

def get_answer_from_document(context: str, question: str) -> str:
    """
    This function uses a pretrained model 'deepset/roberta-base-squad2-distilled' from Hugging Face Transformers
    to answer questions automatically from a given context.

    Args:
        context (str): The context from which the question will be answered.
        question (str): The question that needs to be answered.

    Returns:
        str: The answer to the question based on the context.
    """
    qa_model = AutoModel.from_pretrained('deepset/roberta-base-squad2-distilled')
    qa_pipeline = pipeline('question-answering', model=qa_model)
    result = qa_pipeline({'context': context, 'question': question})
    return result['answer']

# test_function_code --------------------

def test_get_answer_from_document():
    """
    This function tests the get_answer_from_document function.
    It uses a sample context and question, and checks if the function returns a string.
    """
    context = 'This is a context.'
    question = 'What is this?'
    answer = get_answer_from_document(context, question)
    assert isinstance(answer, str), 'The function should return a string.'

# call_test_function_code --------------------

test_get_answer_from_document()