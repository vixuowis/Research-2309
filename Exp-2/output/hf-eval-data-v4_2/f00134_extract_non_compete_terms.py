# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_non_compete_terms(context, question):
    """
    Extract information about the non-compete clause from a legal document.

    Args:
        context (str): The legal document containing the data protection provisions and non-compete clause.
        question (str): The question related to the non-compete clause.

    Returns:
        dict: The result containing the answer.

    Raises:
        ValueError: If the context or question is empty.
    """
    if not context or not question:
        raise ValueError('The context and question must not be empty.')

    # Create a question-answering pipeline with a specific CUAD-trained model
    qa_pipeline = pipeline('question-answering', model='Rakib/roberta-base-on-cuad')

    # Use the pipeline to extract the answer
    return qa_pipeline(question=question, context=context)

# test_function_code --------------------

def test_extract_non_compete_terms():
    print("Testing started.")
    test_context = "The non-compete clause states that the service provider is prohibited from providing similar services to any competitor within a 50-mile radius and during the 1-year period following termination of services."
    test_question = "What are the terms of the non-compete clause?"

    # Test case 1: Correct extraction
    print("Testing case [1/1] started.")
    result = extract_non_compete_terms(test_context, test_question)
    assert 'prohibited from providing similar services to any competitor within a 50-mile radius and during the 1-year period' in result['answer'], f"Test case [1/1] failed: {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_extract_non_compete_terms()