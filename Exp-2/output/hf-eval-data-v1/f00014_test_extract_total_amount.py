def test_extract_total_amount():
    # Test dataset
    question = 'What is the total amount?'
    context = 'Invoice information for order ABC_123\nProduct: Widget A, Quantity: 10, Price: $5 each\nProduct: Widget B, Quantity: 5, Price: $3 each\nProduct: Widget C, Quantity: 15, Price: $2 each\nSubtotal: $75, Tax: $6.38, Total Amount Due: $81.38'
    # Expected output
    expected_output = '$81.38'
    # Call the function with the test dataset
    output = extract_total_amount(question, context)
    # Assert that the output is as expected
    assert output == expected_output, f'Expected {expected_output}, but got {output}'

test_extract_total_amount()