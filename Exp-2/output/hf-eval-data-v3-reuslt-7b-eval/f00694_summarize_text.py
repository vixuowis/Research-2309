# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_text(text: str) -> str:
    """
    Summarizes a given text using the 'philschmid/bart-large-cnn-samsum' model from Hugging Face Transformers.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    
    if len(text) > 1024:
        raise ValueError('The input text is too long for this model.')
        
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    result = summarizer(text, max_length=150, min_length=30, do_sample=False)
    
    return ''.join([text.capitalize() for text in result[0]['summary_text'].split('.')])

# test_function_code --------------------

def test_summarize_text():
    """
    Tests the 'summarize_text' function.
    """
    text1 = 'The customer support service was excellent. All our concerns were attended to promptly by the friendly and knowledgeable staff. ...'
    text2 = 'Jeff: Can I train a 🤗 Transformers model on Amazon SageMaker? Philipp: Sure you can use the new Hugging Face Deep Learning Container. ...'
    assert len(summarize_text(text1)) > 0
    assert len(summarize_text(text2)) > 0
    return 'All Tests Passed'


# call_test_function_code --------------------

test_summarize_text()