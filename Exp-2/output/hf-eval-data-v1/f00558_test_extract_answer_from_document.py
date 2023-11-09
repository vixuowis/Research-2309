def test_extract_answer_from_document():
    question = 'Who is the author of the document?'
    context = 'This document was written by John Doe.'
    answer = extract_answer_from_document(question, context)
    assert answer == 'John Doe', f'Error: Expected John Doe, but got {answer}'

test_extract_answer_from_document()