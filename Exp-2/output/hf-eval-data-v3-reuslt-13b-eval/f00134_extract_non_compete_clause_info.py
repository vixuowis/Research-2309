# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_non_compete_clause_info(context: str, question: str) -> str:
    '''
    Extracts information about a non-compete clause from a legal document with a context related to data protection.

    Args:
        context (str): The legal document from which to extract information.
        question (str): The question related to the non-compete clause.

    Returns:
        str: The extracted answer based on the given context.
    '''

    model = pipeline('question-answering')

    result = model(context=context, 
                   question=question)
    
    return result['answer']

# test_function_code --------------------

def test_extract_non_compete_clause_info():
    '''
    Tests the function extract_non_compete_clause_info.
    '''
    context = 'The data protection provisions set forth in this agreement shall be in effect for a period of 2 years after the termination of services. The non-compete clause states that the service provider is prohibited from providing similar services to any competitor within a 50-mile radius and during the 1-year period following termination of services.'
    question = 'What are the terms of the non-compete clause?'
    assert isinstance(extract_non_compete_clause_info(context, question), str)
    question = 'What is the duration of the non-compete clause?'
    assert isinstance(extract_non_compete_clause_info(context, question), str)
    question = 'What is the radius of the non-compete clause?'
    assert isinstance(extract_non_compete_clause_info(context, question), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_extract_non_compete_clause_info()