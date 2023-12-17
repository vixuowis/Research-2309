# requirements_file --------------------

!pip install -U transformers==4.15.0 torch==1.8.0+cu101 datasets==1.17.0 tokenizers==0.10.3

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def document_eligibility_check(document, question):
    """
    Determines the eligibility of an applicant based on a document and a question.

    Args:
        document (str): The document containing the company's policy.
        question (str): The question regarding an applicant's eligibility.

    Returns:
        bool: True if the applicant is eligible based on the document, else False.

    Raises:
        ValueError: If the document or the question is empty.
    """
    if not document or not question:
        raise ValueError("The document and question cannot be empty.")

    qa_model = pipeline('question-answering', model='tiennvcs/layoutlmv2-large-uncased-finetuned-vi-infovqa')
    answer = qa_model({'question': question, 'context': document})
    return 'no' not in answer['answer'].lower()


# test_function_code --------------------

def test_document_eligibility_check():
    print("Testing started.")
    document = "Our company policy restricts the loan applicant's eligibility to the citizens of United States. The applicant needs to be 18 years old or above and their monthly salary should at least be $4,000. FetchTypeOfYear: 2019."

    # Test case 1: Eligible applicant
    print("Testing case [1/3] started.")
    question_eligible = "Can someone with a monthly salary of $5,000 apply?"
    assert document_eligibility_check(document, question_eligible), "Test case [1/3] failed: The answer should be True for an eligible applicant."

    # Test case 2: Ineligible applicant
    print("Testing case [2/3] started.")
    question_ineligible = "Can anyone with a monthly salary of $3,000 apply?"
    assert not document_eligibility_check(document, question_ineligible), "Test case [2/3] failed: The answer should be False for an ineligible applicant."

    # Test case 3: Empty document or question
    print("Testing case [3/3] started.")
    try:
        document_eligibility_check('', question_ineligible)
        raise AssertionError("Test case [3/3] failed: An exception for empty document or question was expected.")
    except ValueError:
        pass  # Expected exception
    print("Testing finished.")


# call_test_function_line --------------------

test_document_eligibility_check()