# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_customer_question(question, context):
    """
    Answer the customer's query in Korean using the question-answering model.
    
    Parameters:
        question (str): The question asked by the customer in Korean.
        context (str): The relevant context in Korean.

    Returns:
        str: The answer extracted from the context for the given question.
    """
    qa_pipeline = pipeline('question-answering', model='monologg/koelectra-small-v2-distilled-korquad-384')
    answer = qa_pipeline(question=question, context=context)['answer']
    return answer

# test_function_code --------------------

def test_answer_customer_question():
    print("Testing answer_customer_question function.")
    # Example context and question in Korean
    context = "이 프로그램은 한국어 멀티미디어 앱의 고객 문의 사항에 자동으로 답변하는 기능을 제공합니다."
    question = "이 앱의 주요 기능은 무엇입니까?"
    # Expected answer
    expected = "고객 문의 사항에 자동으로 답변하는 기능"

    # Get the answer
    answer = answer_customer_question(question, context)

    # Check if the answer is correct
    assert answer == expected, f"Test failed: expected '{expected}', but got '{answer}'"
    print("Test passed.")

# Run the test
test_answer_customer_question()