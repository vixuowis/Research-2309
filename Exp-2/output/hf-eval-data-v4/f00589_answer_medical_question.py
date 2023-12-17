# requirements_file --------------------

!pip install -U transformers sentencepiece

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def answer_medical_question(context, question):
    """
    This function uses a pre-trained model on biomedical data to answer medical questions.

    Parameters:
    context (str): A string providing the background information related to the medical question.
    question (str): The medical question to be answered.

    Returns:
    dict: Contains the answer to the question based on the provided context.
    """
    # Initialize the question-answering pipeline with the biomedical model
    qa_pipeline = pipeline('question-answering', model='sultan/BioM-ELECTRA-Large-SQuAD2')

    # Use the pipeline to get the answer to the question
    result = qa_pipeline({'context': context, 'question': question})

    return result

# test_function_code --------------------

def test_answer_medical_question():
    print("Testing started.")
    # Provide a context and a sample question for testing
    context = 'Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.'
    question = 'What causes COVID-19?'

    # Testing case
    print("Testing case [1/1] started.")
    answer = answer_medical_question(context, question)
    assert 'coronavirus' in answer['answer'], f"Test case failed: Expected 'coronavirus' in the answer, got {answer['answer']}"
    print("Testing finished.")

# Run the test function
test_answer_medical_question()