def test_classify_inquiry():
    """
    Test the classify_inquiry function.
    """
    test_inquiry = 'I am experiencing difficulty with the installation process of your software.'
    assert classify_inquiry(test_inquiry) == 'technical support'
    test_inquiry = 'I want to buy your product.'
    assert classify_inquiry(test_inquiry) == 'sales'
    test_inquiry = 'I have a problem with my bill.'
    assert classify_inquiry(test_inquiry) == 'billing'

test_classify_inquiry()