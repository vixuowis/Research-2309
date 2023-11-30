# function_import --------------------

from transformers import pipeline

# function_code --------------------

def question_answering_tool(context: str, question: str) -> dict:
    '''
    This function uses a pretrained model from the transformers library to answer questions based on a given context.

    Args:
        context (str): The context from which the answer should be extracted.
        question (str): The question for which an answer is needed.

    Returns:
        dict: A dictionary containing the answer and the score of the answer.
    '''
    
    # Initialize the pipeline
    QA_pipeline = pipeline("question-answering")
    
    # Get the answer and calculate its confidence
    answer = QA_pipeline(
        question=question, 
        context=context)
        
    return {'answer': answer['answer'], 'score': round(answer['score']*100)}


# test_function_code --------------------

def test_question_answering_tool():
    '''
    This function tests the question_answering_tool function.
    '''
    context = 'This is a long document containing company policies, financial details, and team structures.'
    question = 'What are the company policies mentioned in the document?'
    answer = question_answering_tool(context, question)
    assert isinstance(answer, dict)
    assert 'answer' in answer
    assert 'score' in answer

    context = 'The sky is blue and the grass is green.'
    question = 'What color is the sky?'
    answer = question_answering_tool(context, question)
    assert answer['answer'] == 'blue'

    context = 'Python is a popular programming language.'
    question = 'What is Python?'
    answer = question_answering_tool(context, question)
    assert answer['answer'] == 'a popular programming language'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_question_answering_tool()