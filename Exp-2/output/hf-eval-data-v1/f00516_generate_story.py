from transformers import pipeline


def generate_story(starting_phrase):
    """
    This function generates a story based on a starting phrase using the 'decapoda-research/llama-13b-hf' model from Hugging Face Transformers.
    
    Parameters:
    starting_phrase (str): The phrase to start the story with.
    
    Returns:
    str: The generated story.
    """
    # Create a text-generation pipeline using the 'decapoda-research/llama-13b-hf' model
    generator = pipeline('text-generation', model='decapoda-research/llama-13b-hf')
    
    # Generate the story
    generated_text = generator(starting_phrase)[0]['generated_text']
    
    return generated_text