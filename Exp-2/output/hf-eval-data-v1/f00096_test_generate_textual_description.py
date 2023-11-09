# Test function for generate_textual_description
# Uses a sample image and text prompt to test the function
# Asserts that the function returns a non-empty string

def test_generate_textual_description():
    # Sample encoded image and text prompt
    # Note: In a real-world scenario, the encoded image would be obtained from a real image
    encoded_image = torch.rand(1, 3, 224, 224)  # Random tensor as a placeholder
    text = 'A sample text prompt'

    # Call the function with the sample inputs
    generated_text = generate_textual_description(encoded_image, text)

    # Assert that the function returns a non-empty string
    assert isinstance(generated_text, str) and len(generated_text) > 0

    print('Test passed.')

# Run the test function
test_generate_textual_description()