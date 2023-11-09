from transformers import TextGenerationPipeline, Bloom7b1Model


def generate_houseplant_care_tips(prompt: str):
    """
    This function uses the Hugging Face Transformers library to generate a paragraph of text on houseplant care tips.
    It uses the pre-trained 'bigscience/bloom-7b1' model from the Hugging Face model hub.
    
    Args:
    prompt (str): The prompt to feed to the text generation model.
    
    Returns:
    str: The generated paragraph of text.
    """
    # Load the pre-trained model
    model = Bloom7b1Model.from_pretrained('bigscience/bloom-7b1')
    
    # Initialize a TextGenerationPipeline with the loaded model
    text_generator = TextGenerationPipeline(model=model)
    
    # Call the pipeline with the provided prompt
    generated_paragraph = text_generator(prompt)[0]['generated_text']
    
    return generated_paragraph