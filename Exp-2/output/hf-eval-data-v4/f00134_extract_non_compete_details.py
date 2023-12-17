# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_non_compete_details(context, question):
    qa_pipeline = pipeline('question-answering', model='Rakib/roberta-base-on-cuad')
    answer = qa_pipeline(question=question, context=context)
    return answer['answer']

# The function extracts specific details regarding the non-compete clause from the provided context using a question-answering model.

# test_function_code --------------------

def test_extract_non_compete_details():
    print("Testing started.")

    # Test case: Extracting non-compete clause details
    context = "The data protection provisions set forth in this agreement shall be in effect for a period of 2 years after the termination of services. The non-compete clause states that the service provider is prohibited from providing similar services to any competitor within a 50-mile radius and during the 1-year period following termination of services."
    question = "What are the terms of the non-compete clause?"

    expected_answer = 'service provider is prohibited from providing similar services to any competitor within a 50-mile radius and during the 1-year period following termination of services'
    actual_answer = extract_non_compete_details(context, question)

    assert actual_answer == expected_answer, f"Test failed: Expected '{expected_answer}', but got '{actual_answer}'"
    print("Testing finished.")

# Running the test function
test_extract_non_compete_details()
