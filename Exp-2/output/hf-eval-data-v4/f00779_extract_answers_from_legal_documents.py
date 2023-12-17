# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def extract_answers_from_legal_documents(question, context):
    # Load the pre-trained question-answering model from Hugging Face Transformers
    nlp = pipeline('question-answering', model='deepset/deberta-v3-large-squad2', tokenizer='deepset/deberta-v3-large-squad2')
    # Create the input dictionary for the question-answering model
    QA_input = {'question': question, 'context': context}
    # Use the model to find answers to the legal question
    result = nlp(QA_input)
    return result

# test_function_code --------------------

def test_extract_answers_from_legal_documents():
    print("Testing started.")
    sample_context = "The party in breach shall be liable to pay the penalty as described in clause 4.2 of the contract."

    # Test case 1: Extracting a known answer
    question1 = "What is the penalty clause number for a breach?"
    result1 = extract_answers_from_legal_documents(question1, sample_context)
    assert '4.2' in result1['answer'], f"Test case failed: Expected '4.2', got {result1['answer']}"

    # Test case 2: Handling complex legal terminology
    question2 = "Who is liable when a breach occurs?"
    result2 = extract_answers_from_legal_documents(question2, sample_context)
    assert 'The party in breach' in result2['answer'], f"Test case failed: Expected 'The party in breach', got {result2['answer']}"

    print("Testing finished.")

# Run the test function
test_extract_answers_from_legal_documents()