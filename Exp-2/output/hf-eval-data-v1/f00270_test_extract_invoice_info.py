def test_extract_invoice_info():
    """
    This function tests the 'extract_invoice_info' function.
    It uses a sample invoice image and checks if the function returns the correct information.
    """
    image_path = 'path/to/sample_invoice.jpg'
    result = extract_invoice_info(image_path)
    assert isinstance(result, dict), 'The function should return a dictionary.'
    assert 'total_amount' in result, 'The total amount should be in the result.'
    assert 'date_of_invoice' in result, 'The date of the invoice should be in the result.'
    assert 'service_provider' in result, 'The name of the service provider should be in the result.'

test_extract_invoice_info()