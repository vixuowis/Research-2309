def test_generate_welcome_email():
    """
    This function tests the 'generate_welcome_email' function.
    It uses a fixed seed text and checks if the generated email starts with the seed text.
    """
    # Define the seed text
    seed_text = 'Welcome to the company,'
    
    # Generate the email using the 'generate_welcome_email' function
    generated_email = generate_welcome_email(seed_text)
    
    # Check if the generated email starts with the seed text
    assert generated_email.startswith(seed_text), 'The generated email does not start with the seed text.'