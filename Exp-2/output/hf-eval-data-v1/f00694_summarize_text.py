from transformers import pipeline


def summarize_text(feedback):
    """
    This function uses the Hugging Face Transformers library to summarize a given text.
    It uses the 'philschmid/bart-large-cnn-samsum' model for summarization.
    
    Parameters:
    feedback (str): The text to be summarized.
    
    Returns:
    str: The summarized text.
    """
    # Create a summarization model using the pipeline function
    summarizer = pipeline('summarization', model='philschmid/bart-large-cnn-samsum')
    
    # Pass the customer feedback document as input to the model
    summary = summarizer(feedback)
    
    return summary[0]['summary_text']