def test_generate_summary():
    # Test the generate_summary function
    # Define a test case
    text = 'Cryptocurrencies have become exceedingly popular among investors seeking higher returns and diversification in their portfolios. However, investing in these digital currencies carries several inherent risks.'
    # Call the function with the test case
    summary = generate_summary(text)
    # Assert that the function returns a string
    assert isinstance(summary, str)
    # Assert that the summary is not empty
    assert len(summary) > 0

test_generate_summary()