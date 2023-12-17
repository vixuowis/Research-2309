# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def find_textbook_answer(question, textbook_content):
    """
    This function uses a pre-trained DistilBERT model to find the answer to a question 
    from the given textbook content.

    Args:
    question (str): A question phrase to be answered.
    textbook_content (str): The context from the textbook where the answer may be found.

    Returns:
    str: The answer extracted from the textbook content.
    """
    # Create the question-answering model pipeline
    qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    
    # Get the answer to the question from textbook content
    result = qa_model(question=question, context=textbook_content)
    
    # Return the answer found
    return result['answer']

# test_function_code --------------------

def test_find_textbook_answer():
    print("Testing started.")
    
    # Test setup
    textbook_content = "Mitochondria are the energy factories of the cell. " \
                       "They convert energy from food molecules into a useable form " \
                       "known as adenosine triphosphate (ATP)."
    
    # Test case 1: Straightforward question
    print("Testing case [1/3] started.")
    answer1 = find_textbook_answer(question="What is the function of mitochondria in a cell?", context=textbook_content)
    assert answer1 == "They convert energy from food molecules into a useable form known as adenosine triphosphate (ATP).", f"Test case [1/3] failed: Expected answer not found, got {answer1}"
    print("Test case [1/3] passed.")

    # Test case 2: Question with a specific term
    print("Testing case [2/3] started.")
    answer2 = find_textbook_answer(question="What do mitochondria produce?", context=textbook_content)
    assert answer2 == "adenosine triphosphate (ATP)", f"Test case [2/3] failed: Expected 'adenosine triphosphate (ATP)', got {answer2}"
    print("Test case [2/3] passed.")

    # Test case 3: Question unrelated to the textbook content
    print("Testing case [3/3] started.")
    answer3 = find_textbook_answer(question="Who is the president of the United States?", context=textbook_content)
    assert answer3 == "No valid answer", f"Test case [3/3] failed: Expected 'No valid answer', got {answer3}"
    print("Test case [3/3] passed.")

    print("Testing finished.")

# Run the test function
test_find_textbook_answer()