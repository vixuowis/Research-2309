def test_get_max_goal_scorer():
    # Test dataset
    question = "What player scored the most goals?"
    table = "Player,Goals\nA,2\nB,3\nC,1"
    # Expected output
    expected_output = 'B'
    # Get the function output
    function_output = get_max_goal_scorer(question, table)
    # Assert that the function output is close to the expected output
    assert function_output == expected_output, f'Expected {expected_output}, but got {function_output}'

test_get_max_goal_scorer()