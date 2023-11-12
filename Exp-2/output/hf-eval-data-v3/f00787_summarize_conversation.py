# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_conversation(conversation):
    """
    Summarize a conversation using the Hugging Face Transformers library.

    Args:
        conversation (str): The conversation to be summarized.

    Returns:
        str: The summarized conversation.
    """
    summarizer = pipeline('summarization', model='lidiya/bart-large-xsum-samsum')
    summary = summarizer(conversation)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_conversation():
    """
    Test the summarize_conversation function.
    """
    conversation1 = '''Hannah: Hey, do you have Betty's number?
Amanda: Lemme check
Amanda: Sorry, can't find it.
Amanda: Ask Larry
Amanda: He called her last time we were at the park together
Hannah: I don't know him well
Amanda: Don't be shy, he's very nice
Hannah: If you say so..
Hannah: I'd rather you texted him
Amanda: Just text him 🙂
Hannah: Urgh.. Alright
Hannah: Bye
Amanda: Bye bye
'''
    assert isinstance(summarize_conversation(conversation1), str)
    conversation2 = 'A: Hi
B: Hello
A: How are you?
B: Good, and you?
A: Good, thanks!'
    assert isinstance(summarize_conversation(conversation2), str)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_conversation()