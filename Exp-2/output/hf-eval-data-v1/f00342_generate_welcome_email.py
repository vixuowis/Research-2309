from transformers import pipeline


def generate_welcome_email(seed_text):
    """
    This function generates a welcome email for a new employee joining the company.
    It uses the 'lewtun/tiny-random-mt5' model from Hugging Face Transformers for text generation.
    
    Parameters:
    seed_text (str): The seed text to start the email.
    
    Returns:
    str: The generated email.
    """
    # Create a text generation model using the pipeline function from the transformers library
    text_generator = pipeline('text-generation', model='lewtun/tiny-random-mt5')
    
    # Generate the email using the text generation model and the provided seed text
    generated_email = text_generator(seed_text, max_length=150)
    
    return generated_email[0]['generated_text']