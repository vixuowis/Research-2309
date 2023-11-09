def test_generate_marketing_content():
    """
    Test the generate_marketing_content function.
    """
    prompt = 'Introducing our new line of eco-friendly kitchenware:'
    generated_content = generate_marketing_content(prompt)
    assert isinstance(generated_content, str), 'The output should be a string.'
    assert len(generated_content) > len(prompt), 'The generated content should be longer than the prompt.'

test_generate_marketing_content()