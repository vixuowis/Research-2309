# requirements_file --------------------

!pip install -U transformers torch datasets tokenizers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_answer_from_medical_document(question, document_text):
    """
    This function uses a pre-trained BERT model to extract answers from a medical document based on a given question.
    :param question: str, a question posed by the doctor
    :param document_text: str, the medical document text
    :return: str, the answer extracted from the document
    """
    # Initialize the question-answering pipeline with the specified model
    qa_pipeline = pipeline('question-answering', model='bigwiz83/sapbert-from-pubmedbert-squad2')
    # Use the pipeline to get the answer from the document
    answer = qa_pipeline({'context': document_text, 'question': question})
    return answer['answer']

# test_function_code --------------------

def test_extract_answer_from_medical_document():
    print("Testing started.")
    # A sample medical document text
    document_text = "Pembrolizumab shows promise for the treatment of advanced melanoma."
    question = "What is Pembrolizumab used for?"
    # Expected answer
    expected_answer = "treatment of advanced melanoma"
    # Extract the answer using the function
    answer = extract_answer_from_medical_document(question, document_text)
    # Check if the function gives the correct answer
    assert answer == expected_answer, f"Test failed: incorrect answer {answer}"
    print("Testing finished.")

# Run the test function
test_extract_answer_from_medical_document()