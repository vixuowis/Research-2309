from transformers import pipeline


def fill_mask(masked_text):
    """
    This function fills the mask in a given text using a multilingual model.
    
    Parameters:
    masked_text (str): The text with a [MASK] token.
    
    Returns:
    str: The text with the [MASK] token replaced by the predicted word.
    """
    # Import the pipeline function from the transformers package
    unmasker = pipeline('fill-mask', model='distilbert-base-multilingual-cased')
    
    # Call the unmasker function with the provided text that has a [MASK] token in it
    # The masked language model will then suggest a word that fits the context
    result = unmasker(masked_text)
    
    return result[0]['sequence']