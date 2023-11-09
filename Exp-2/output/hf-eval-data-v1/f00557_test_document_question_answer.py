# Test function for document_question_answer
# Uses a sample question and scanned document text
def test_document_question_answer():
    # Sample question and scanned document text
    question = 'What is the title of the document?'
    scanned_document_text = 'This is a sample document. The title is Sample Document.'

    # Call the function with the sample inputs
    output = document_question_answer(question, scanned_document_text)

    # Assert that the output is not None
    assert output is not None

    # Print the output for visual inspection
    print(output)

# Call the test function
test_document_question_answer()