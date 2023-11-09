from transformers import pipeline

def generate_story(prompt: str, max_length: int = 500) -> str:
    """
    This function generates a story based on the given prompt using the GPT-J 6B model.
    
    Parameters:
    prompt (str): The prompt for the story.
    max_length (int): The maximum length of the story. Default is 500.
    
    Returns:
    str: The generated story.
    """
    # Load the GPT-J 6B model
    text_generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')
    
    # Generate the story
    story_output = text_generator(prompt, max_length=max_length)
    
    # Extract the story from the output
    story = story_output[0]['generated_text']
    
    return story