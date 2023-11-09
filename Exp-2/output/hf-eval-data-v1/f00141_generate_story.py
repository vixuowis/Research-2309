from transformers import pipeline


def generate_story(prompt):
    """
    This function generates a short story based on a given prompt using the LLaMA-7B language model.
    
    Parameters:
    prompt (str): The initial prompt for the story.
    
    Returns:
    str: The generated story.
    """
    # Create a text generation model using the pipeline function
    story_generator = pipeline('text-generation', model='decapoda-research/llama-7b-hf')
    
    # Use the generated model to create a story based on the given prompt
    story = story_generator(prompt)
    
    # The result is a list of generated texts, and we can take the first generated text as the final output
    return story[0]['generated_text']