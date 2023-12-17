# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_chat(conversation):
    """
    Summarize a chat conversation using a Hugging Face Transformer model.

    Args:
        conversation (str): A string containing the chat conversation to summarize.

    Returns:
        str: A summarized version of the conversation.

    Raises:
        ValueError: If the conversation is empty.
    """
    if not conversation:
        raise ValueError('The conversation cannot be empty')
    summarizer = pipeline('summarization', model='lidiya/bart-large-xsum-samsum')
    summary = summarizer(conversation)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_chat():
    print("Testing started.")
    conversation = '''Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nAmanda: Sorry, can't find it.\nAmanda: Ask Larry\nAmanda: He called her last time we were at the park together\nHannah: I don't know him well\nAmanda: Don't be shy, he's very nice\nHannah: If you say so..\nHannah: I'd rather you texted him\nAmanda: Just text him \U0001F642\nHannah: Urgh.. Alright\nHannah: Bye\nAmanda: Bye bye\n'''

    # Testing case 1: Normal conversation
    print("Testing case [1/2] started.")
    expected_summary = 'Hannah asked Amanda for Betty's number, but Amanda couldn't find it and suggested Hannah ask Larry.'
    assert summarize_chat(conversation) == expected_summary, "Test case [1/2] failed: Summary did not match expected output."

    # Testing case 2: Empty conversation
    print("Testing case [2/2] started.")
    empty_conversation = ''
    try:
        summarize_chat(empty_conversation)
        assert False, "Test case [2/2] failed: Expected ValueError for empty conversation."
    except ValueError as e:
        assert str(e) == 'The conversation cannot be empty', "Test case [2/2] failed: Incorrect ValueError message."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_chat()