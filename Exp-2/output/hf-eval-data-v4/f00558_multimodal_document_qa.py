# requirements_file --------------------

!pip install -U transformers>=4.0.0

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

# function_code --------------------

def multimodal_document_qa(model_checkpoint, question, context):
    # Initialize the tokenizer and the document question answering model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)

    # Tokenize the input data
    inputs = tokenizer(question, context, return_tensors='pt')

    # Run the document-question-answering model
    outputs = model(**inputs)

    # Extract the indices for the start and end of the answer
    ans_start = outputs.start_logits.argmax()  # index of the start position
    ans_end = outputs.end_logits.argmax()      # index of the end position

    # Decode the answer from the tokenized format
    answer = tokenizer.decode(inputs['input_ids'][0][ans_start : ans_end + 1])

    return answer

# test_function_code --------------------

def test_multimodal_document_qa():
    print("Testing multimodal_document_qa function.")
    # Sample data (here you should use real examples)
    model_checkpoint = 'L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023'
    question = 'What is the main topic of the document?'
    context = 'This document provides an overview of the multimodal question answering with documents.'

    # Test the function
    expected_answer = 'multimodal question answering'
    actual_answer = multimodal_document_qa(model_checkpoint, question, context)

    # Validate the result
    assert actual_answer == expected_answer, f"Test failed: Expected answer was '{expected_answer}', but got '{actual_answer}'"
    print("Test passed successfully.")

# Run the test function
test_multimodal_document_qa()