# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_conversation(conversation_text):
    """Summarize a conversation using a pre-trained NLP model.

    Args:
        conversation_text (str): The conversation text to summarize.

    Returns:
        str: The summary of the conversation.

    Raises:
        ValueError: If the input conversation_text is not a string.
    """
    if not isinstance(conversation_text, str):
        raise ValueError('The conversation_text must be a string.')

    summarizer = pipeline('summarization', model='philschmid/distilbart-cnn-12-6-samsum')
    summary = summarizer(conversation_text)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_conversation():
    print('Testing started.')
    conversation = 'Anna: We discussed marketing budget. Tom: Funds to social media. Sarah: SEO improvement. Anna: Content creation. Tom: Strategy revision. Sarah: New KPIs.'

    # Test case 1: Valid input
    print('Testing case [1/1] started.')
    summary = summarize_conversation(conversation)
    assert summary, f'Test case [1/1] failed: No summary generated.'
    print('Testing finished.')

# call_test_function_line --------------------

test_summarize_conversation()