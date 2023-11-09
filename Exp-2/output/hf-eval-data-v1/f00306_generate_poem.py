from transformers import pipeline


def generate_poem(prompt):
    """
    This function generates a poem based on a given prompt using the 'sshleifer/tiny-gpt2' model.
    The model is a smaller version of the GPT-2 model designed for faster inference and lower resource usage.
    
    Parameters:
    prompt (str): The initial string to base the poem on.
    
    Returns:
    str: The generated poem.
    """
    # Create a text generation pipeline with the 'sshleifer/tiny-gpt2' model
    nlp = pipeline('text-generation', model='sshleifer/tiny-gpt2')
    # Generate the poem
    result = nlp(prompt)
    # Return the generated poem
    return result[0]['generated_text']