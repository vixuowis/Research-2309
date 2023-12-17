# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_information_from_document(question, context):
    """
    Extracts information by answering a question based on the provided document context using a pre-trained model.

    Parameters:
    question (str): The question to be answered.
    context (str): The context from the document to find the answer.

    Returns:
    str: The answer extracted from the document.
    """
    # Load the pre-trained question-answering model
    doc_qa = pipeline('question-answering', model='Sayantan1993/layoutlmv2-base-uncased_finetuned_docvqa')
    # Get the answer by providing the question and context to the model
    answer = doc_qa(question=question, context=context)
    return answer

# test_function_code --------------------

def test_extract_information_from_document():
    print("Testing started.")

    # Test case 1
    question1 = 'When does the contract expire?'
    context1 = 'The contract shall commence on the date hereof and shall continue in effect until terminated by either party with a 30-day notice.'
    expected_answer1 = 'either party with a 30-day notice'
    answer1 = extract_information_from_document(question1, context1)
    assert expected_answer1.lower() in answer1['answer'].lower(), f"Test case [1/3] failed: {answer1}"

    # Test case 2
    question2 = 'What is the governing law of the contract?'
    context2 = 'This contract shall be governed by and construed in accordance with the laws of the State of California.'
    expected_answer2 = 'the laws of the State of California'
    answer2 = extract_information_from_document(question2, context2)
    assert expected_answer2.lower() in answer2['answer'].lower(), f"Test case [2/3] failed: {answer2}"

    # Test case 3
    question3 = 'Who is the service provider?'
    context3 = 'The Service Provider shall be XYZ Corporation.'
    expected_answer3 = 'XYZ Corporation'
    answer3 = extract_information_from_document(question3, context3)
    assert expected_answer3.lower() in answer3['answer'].lower(), f"Test case [3/3] failed: {answer3}"
    print("Testing finished.")

# Run the test function
test_extract_information_from_document()