def test_get_answer_from_pdf():
    """
    This function tests the get_answer_from_pdf function.
    It uses a sample image URL and a question, and checks if the function returns a result.
    """
    # Define a sample image URL and a question.
    image_url = 'https://templates.invoicehome.com/invoice-template-us-neat-750px.png'
    question = 'What is the invoice number?'
    # Call the function with the sample data.
    result = get_answer_from_pdf(image_url, question)
    # Check if the function returns a result.
    assert result is not None, 'The function did not return a result.'
    assert isinstance(result, str), 'The function did not return a string.'

test_get_answer_from_pdf()