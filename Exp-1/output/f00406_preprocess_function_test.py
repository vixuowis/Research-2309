from f00406_preprocess_function import *
def test_preprocess_function():
    # Define test cases
    test_cases = [
        {
            'input': 'Hello',
            'output': 'World'
        },
        {
            'input': 'Goodbye',
            'output': 'Farewell'
        }
    ]

    # Apply the preprocess function to each test case
    for test_case in test_cases:
        preprocessed_test_case = preprocess_function(test_case)

        # Assert the expected preprocessed example
        assert preprocessed_test_case['input'] == test_case['input'].lower()
        assert preprocessed_test_case['output'] == test_case['output'].upper()

    print('All test cases pass')
