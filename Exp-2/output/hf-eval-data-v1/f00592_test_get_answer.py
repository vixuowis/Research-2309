def test_get_answer():
    # Test dataset
    context = 'Inflation is an increase in the general price level of goods and services in an economy over time.'
    question = 'What is inflation?'
    # Expected answer
    expected_answer = 'an increase in the general price level of goods and services in an economy over time'
    # Get the actual answer
    actual_answer = get_answer(question, context)
    # Assert that the actual answer is close to the expected answer
    assert actual_answer.lower() in expected_answer.lower(), f'Expected {expected_answer}, but got {actual_answer}'

test_get_answer()