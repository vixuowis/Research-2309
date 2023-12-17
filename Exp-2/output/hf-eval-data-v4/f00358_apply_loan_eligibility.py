# requirements_file --------------------

!pip install -U transformers, torch, datasets, tokenizers

# function_import --------------------

from transformers import pipeline


# function_code --------------------

def apply_loan_eligibility(document, question):
    """
    Check if an applicant is eligible to apply for a loan based on company policy.
    
    Args:
        document (str): The company policy document.
        question (str): The question about loan application eligibility.
    
    Returns:
        str: 'Yes' if eligible, 'No' otherwise.
    """
    # Initialize the Hugging Face question-answering pipeline
    qa_model = pipeline('question-answering', model='tiennvcs/layoutlmv2-large-uncased-finetuned-vi-infovqa')
    # Use the model to answer the question based on the document
    answer = qa_model({'question': question, 'context': document})['answer']
    # Check if the answer is 'Yes' or 'No'
    return 'Yes' if answer.strip().lower() in ['yes', 'eligible'] else 'No'

# test_function_code --------------------

def test_apply_loan_eligibility():
    print("Testing started.")
    document = "Our company policy restricts the loan applicant's eligibility to the citizens of United States. The applicant needs to be 18 years old or above and their monthly salary should at least be $4,000. FetchTypeOfYear: 2019."
    # Test case 1: Applicant with a salary of $3,000
    question = "Can anyone with a monthly salary of $3,000 apply?"
    print("Testing case [1/1] started.")
    assert apply_loan_eligibility(document, question) == 'No', f"Test case [1/1] failed: Expected 'No' but got {apply_loan_eligibility(document, question)}"
    print("Testing finished.")

# Run the test function
test_apply_loan_eligibility()