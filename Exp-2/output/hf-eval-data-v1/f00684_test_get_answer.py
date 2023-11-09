def test_get_answer():
    # Test dataset
    question = 'What is the capital of France?'
    context = 'France is a country in Europe. Its capital is Paris.'
    # Expected answer
    expected_answer = 'Paris'
    # Get the answer from the function
    answer = get_answer(question, context)
    # Assert that the function's answer is close to the expected answer
    assert answer == expected_answer, f'Expected {expected_answer}, but got {answer}'

test_get_answer()