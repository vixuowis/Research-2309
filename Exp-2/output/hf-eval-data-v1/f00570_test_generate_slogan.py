def test_generate_slogan():
    # Set the API key
    api_key = "..."

    # Generate a slogan
    slogan = generate_slogan(api_key)

    # Check if the slogan is a string
    assert isinstance(slogan, str), "The slogan should be a string"

    # Check if the slogan is not empty
    assert len(slogan) > 0, "The slogan should not be empty"

    print("All tests passed.")

test_generate_slogan()