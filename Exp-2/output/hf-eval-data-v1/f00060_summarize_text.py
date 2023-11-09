from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer


def summarize_text(text):
    """
    This function uses the BigBirdPegasusForConditionalGeneration model from Hugging Face Transformers to generate a summary of a long article.
    
    Parameters:
    text (str): The long article to be summarized.
    
    Returns:
    str: The summary of the article.
    """
    # Instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/bigbird-pegasus-large-bigpatent')
    
    # Load the pre-trained BigBird Pegasus model for text summarization
    model = BigBirdPegasusForConditionalGeneration.from_pretrained('google/bigbird-pegasus-large-bigpatent')
    
    # Provide the long article as input to the tokenizer, which returns a dictionary of input tensors
    inputs = tokenizer(text, return_tensors='pt')
    
    # Use the model's generate() method to create a summary of the article from the input tensors
    prediction = model.generate(**inputs)
    
    # Decode the generated tokens back into a summary text
    summary = tokenizer.batch_decode(prediction)[0]
    
    return summary