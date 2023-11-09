def test_classify_text():
    # Define a test text message and a list of possible categories
    text_message = 'I spent hours in the kitchen trying a new recipe.'
    categories = ['travel', 'cooking', 'dancing']

    # Call the classify_text function with the test data
    result = classify_text(text_message, categories)

    # Assert that the result is not None (i.e., the function should always return a category)
    assert result is not None

    # Assert that the result is in the list of possible categories
    assert result in categories

    # Print the result for debugging purposes
    print(f'The category of the text message is: {result}')

# Call the test function
test_classify_text()