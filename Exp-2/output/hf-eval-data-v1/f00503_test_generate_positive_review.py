def test_generate_positive_review():
    """
    This function tests the generate_positive_review function.
    It uses a sample book summary and checks if the output is a string.
    """
    # Sample book summary
    book_summary = 'The book is about the adventures of a young wizard and his friends at a magic school.'
    # Generate the positive review
    positive_review = generate_positive_review(book_summary)
    # Check if the output is a string
    assert isinstance(positive_review, str), 'The output should be a string.'
    # Check if the output is not empty
    assert positive_review != '', 'The output should not be empty.'

test_generate_positive_review()