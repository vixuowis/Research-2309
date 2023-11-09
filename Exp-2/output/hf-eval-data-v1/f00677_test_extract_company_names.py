def test_extract_company_names():
    # Test the extract_company_names function with a sample text
    text = 'I love AutoTrain'
    company_names = extract_company_names(text)
    assert 'AutoTrain' in company_names, 'AutoTrain should be identified as a company name'

    # Test the function with another sample text
    text = 'I use Google and Microsoft products'
    company_names = extract_company_names(text)
    assert 'Google' in company_names, 'Google should be identified as a company name'
    assert 'Microsoft' in company_names, 'Microsoft should be identified as a company name'

# Run the test function
test_extract_company_names()