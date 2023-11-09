# Test function for generate_promotional_poster

def test_generate_promotional_poster():
    # Define the prompt and negative prompt for the test
    prompt = 'A promotional poster for a new line of summer clothing featuring happy people wearing the clothes, with a sunny beach background, clear blue sky, and palm trees. Image dimensions should be poster-sized, high-resolution, and vibrant colors.'
    negative_prompt = 'winter, snow, cloudy, low-resolution, dull colors, indoor, mountain'
    # Call the function with the test prompt and negative prompt
    result = generate_promotional_poster(prompt, negative_prompt)
    # Assert that the result is not None
    assert result is not None
    # Assert that the result is an instance of the expected type
    assert isinstance(result, type(expected_output))

# Call the test function
test_generate_promotional_poster()