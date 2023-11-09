def test_extract_answer():
    """
    This function tests the 'extract_answer' function.
    """
    manual_content = 'This is a test manual. To perform a factory reset, press the reset button for 5 seconds.'
    question = 'How to perform a factory reset on the product?'
    expected_answer = 'press the reset button for 5 seconds'

    assert extract_answer(manual_content, question) == expected_answer