# requirements_file --------------------

!pip install -U transformers onnx

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_customer_questions(context, question):
    """
    Answer customer questions based on the provided context.

    :param context: str, the text containing the information.
    :param question: str, the question posed by the customer.
    :return: str, the answer to the question derived from the context.
    """
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')
    answer = qa_pipeline({'context': context, 'question': question})
    return answer['answer']

# test_function_code --------------------

def test_answer_customer_questions():
    print("Testing started.")
    context = "Hugging Face is a technology company based in New York and Paris focused on natural language processing technologies."
    question = "Where is Hugging Face based?"

    # Test case 1: Check if the function extracts the correct answer
    print("Testing case [1/1] started.")
    expected_answer = "New York and Paris"
    actual_answer = answer_customer_questions(context, question)
    assert actual_answer == expected_answer, f"Test case [1/1] failed: Expected '{{expected_answer}}', got '{{actual_answer}}'"
    print("Testing finished.")

# Run the test function
test_answer_customer_questions()