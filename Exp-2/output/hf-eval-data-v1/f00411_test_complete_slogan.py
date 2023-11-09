def test_complete_slogan():
    """
    This function tests the complete_slogan function.
    It uses a test dataset and assert to ensure the function works as expected.
    """
    # Test dataset
    test_slogans = [
        'Customer satisfaction is our top <mask>.',
        'Your happiness is our <mask>.',
        'We strive for <mask> in all we do.',
    ]
    # Expected outputs
    expected_outputs = [
        'Customer satisfaction is our top priority.',
        'Your happiness is our goal.',
        'We strive for excellence in all we do.',
    ]
    # Test the function with the test dataset
    for i, slogan in enumerate(test_slogans):
        assert complete_slogan(slogan) in expected_outputs, f'Error in test {i+1}'

test_complete_slogan()