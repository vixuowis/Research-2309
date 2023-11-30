# function_import --------------------

from transformers import AutoModel, pipeline

# function_code --------------------

def get_answer_from_document(context: str, question: str) -> str:
    """
    This function uses a pretrained model 'deepset/roberta-base-squad2-distilled' to answer questions automatically from a given context.

    Args:
        context (str): The context from which the question will be answered.
        question (str): The question that needs to be answered.

    Returns:
        str: The answer to the question based on the context.

    Raises:
        Exception: If the model or tokenizer is not properly initialized.
    """
    
    # initialize a pipeline with pretrained model 'deepset/roberta-base-squad2-distilled'
    try:
        answer_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2-distilled")
    
    except Exception as e:
        
        print(e)
        raise Exception("Pipeline initialization failed.")

    # get the answers to a question from context using the pipeline created earlier.
    try:
        answer = answer_pipeline({
            'question': question,
            'context': context
        })['answer']
    
    except Exception as e:
        
        print(e)
        raise Exception("Getting answers failed.")

# test_function_code --------------------

def test_get_answer_from_document():
    """
    This function tests the get_answer_from_document function with some test cases.
    """
    context = 'This is a context.'
    question = 'What is this?'
    assert get_answer_from_document(context, question) is not None

    context = 'The sky is blue.'
    question = 'What color is the sky?'
    assert get_answer_from_document(context, question) is not None

    context = 'Python is a programming language.'
    question = 'What is Python?'
    assert get_answer_from_document(context, question) is not None

    return 'All Tests Passed'


# call_test_function_code --------------------

test_get_answer_from_document()