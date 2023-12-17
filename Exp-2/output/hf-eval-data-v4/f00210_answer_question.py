# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_question(question, context):
    # Answer the provided question based on the given context
    # using the Hugging Face Transformers pipeline.
    # Args:
    #     question (str): The question to answer.
    #     context (str): The context within which to find the answer.
    # Returns:
    #     str: The answer to the question.
    qa_model = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')
    result = qa_model({'question': question, 'context': context})
    return result['answer']

# test_function_code --------------------

def test_answer_question():
    print("Testing started.")

    # Test case 1: Known context about Stockholm
    print("Testing case [1/1] started.")
    question = 'What is the capital of Sweden?'
    context = 'Stockholm is the beautiful capital of Sweden, which is known for its high living standards and great attractions.'
    expected_answer = 'Stockholm'
    answer = answer_question(question, context)
    assert answer == expected_answer, f"Test case [1/1] failed: Expected {expected_answer}, got {answer}"
    print("Testing finished.")

# Run the test function
test_answer_question()