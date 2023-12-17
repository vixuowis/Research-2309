# requirements_file --------------------

!pip install -U transformers onnxruntime

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def question_answering_system(context, question):
    # Instantiate the question-answering pipeline using the specified model
    qa_pipeline = pipeline('question-answering', model='philschmid/distilbert-onnx')

    # Use the pipeline to get the answer for the given question based on the provided context
    answer = qa_pipeline({'context': context, 'question': question})

    # Return the answer obtained from the model
    return answer

# test_function_code --------------------

def test_question_answering_system():
    print("Testing started.")
    context = 'The law stipulates that all contracts must be reviewed by a lawyer before finalization.'
    question = 'Who must review contracts before finalization?'

    # Test case: Check if the answer is 'a lawyer'
    print("Testing case [1/1] started.")
    answer = question_answering_system(context, question)
    assert 'a lawyer' in answer['answer'], f"Test case failed: Expected answer 'a lawyer', but got {answer['answer']}"
    print("Testing case [1/1] succeeded.")
    print("Testing finished.")

# Run the test function
test_question_answering_system()