from transformers import pipeline


def detect_gpt2_generated_text(text):
    """
    This function uses the 'roberta-base-openai-detector' model from the transformers library to detect if the provided text was generated by a GPT-2 model.
    
    Parameters:
    text (str): The text to be analyzed.
    
    Returns:
    dict: The prediction result from the model.
    """
    # Load the 'roberta-base-openai-detector' model
    pipe = pipeline('text-classification', model='roberta-base-openai-detector')
    
    # Use the model to predict if the text was generated by GPT-2
    prediction = pipe(text)
    
    return prediction