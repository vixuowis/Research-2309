def test_translate_color_to_italian():
    """
    This function tests the translate_color_to_italian function by comparing the output for a known input to the expected output.
    """
    # Test data
    test_data = ['red', 'blue', 'green']
    expected_output = ['rosso', 'blu', 'verde']

    # Test the function
    for i, color in enumerate(test_data):
        assert translate_color_to_italian(color) == expected_output[i]

test_translate_color_to_italian()