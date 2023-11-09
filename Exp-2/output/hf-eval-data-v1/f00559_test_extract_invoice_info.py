def test_extract_invoice_info():
    '''
    This function tests the extract_invoice_info function.
    It uses a sample invoice image and checks if the function returns the correct answers.
    '''
    image = 'test_invoice_image.jpg' # replace with path to your test invoice image
    answers = extract_invoice_info(image)
    expected_answers = ['100.00', 'INV123', '2023-03-30'] # replace with the expected answers
    for answer, expected_answer in zip(answers, expected_answers):
        assert answer in expected_answer, f'Expected {expected_answer}, but got {answer}'

test_extract_invoice_info()