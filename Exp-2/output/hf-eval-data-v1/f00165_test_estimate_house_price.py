def test_estimate_house_price():
    # Test dataset
    test_data = {'feat_1': [1], 'feat_2': [2], 'feat_3': [3], 'feat_n': [4]}

    # Expected output
    expected_output = [123456]

    # Call the function with the test dataset
    output = estimate_house_price(test_data)

    # Since we are dealing with estimates, we cannot compare numbers strictly.
    # Therefore, we check if the output is within a reasonable range.
    assert 100000 <= output[0] <= 150000, f'Expected output within range 100000-150000, but got {output[0]}'

# Run the test function
test_estimate_house_price()