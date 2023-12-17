# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def get_answer(question, context):
    """
    This function takes a question and a context and returns the answer to the question based on the given context.
    
    Parameters:
        question (str): The question to be answered.
        context (str): The text containing the answer.
    
    Returns:
        dict: A dictionary containing the answer and other details.
    """
    # Initialize the QA pipeline with the specified RoBERTa model
    qa_pipeline = pipeline('question-answering', model='deepset/roberta-large-squad2')
    
    # Prepare the dictionary for question and context
    question_context = {'question': question, 'context': context}
    
    # Pass the dictionary to the model and get the answer
    answer = qa_pipeline(question_context)
    return answer

# test_function_code --------------------

def test_get_answer():
    print("Testing get_answer function.")
    
    # Test case 1: Known context
    question1 = 'What is the capital of Germany?'
    context1 = 'Berlin is the capital of Germany.'
    print("Test case 1: Known context")
    answer1 = get_answer(question1, context1)
    assert answer1['answer'] == 'Berlin', f"Test case failed: Expected 'Berlin', got {answer1['answer']}"
    print("Test case 1 passed.")
    
    # Test case 2: Known context with different phrasing
    question2 = 'Which city is the German capital?'
    context2 = 'The capital of Germany is Berlin.'
    print("Test case 2: Known context with different phrasing")
    answer2 = get_answer(question2, context2)
    assert answer2['answer'] == 'Berlin', f"Test case failed: Expected 'Berlin', got {answer2['answer']}"
    print("Test case 2 passed.")
    
    # Test case 3: Complex context
    question3 = 'Who was the first Chancellor of Germany?'
    context3 = 'Germany's first Chancellor was Otto von Bismarck.'
    print("Test case 3: Complex context")
    answer3 = get_answer(question3, context3)
    assert 'Bismarck' in answer3['answer'], f"Test case failed: Expected 'Bismarck' in the answer, got {answer3['answer']}"
    print("Test case 3 passed.")
    print("Testing of get_answer function finished.")