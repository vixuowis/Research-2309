# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_conversation(conversation):
    """
    Summarize a chat conversation using a pre-trained model from Hugging Face Transformers.
    
    Args:
    - conversation (str): A string containing the chat conversation to be summarized.
    
    Returns:
    - str: A summary of the conversation.
    """
    # Initialize the summarizer pipeline with the specified model
    summarizer = pipeline('summarization', model='lidiya/bart-large-xsum-samsum')
    
    # Generate the summary for the given conversation
    summary = summarizer(conversation)
    
    # Return the first summary text from the result
    return summary[0]["summary_text"]

# test_function_code --------------------

def test_summarize_conversation():
    print("Testing started.")
    conversation = """Hannah: Hey, do you have Betty's number?
Amanda: Lemme check
Amanda: Sorry, can't find it.
Amanda: Ask Larry
Amanda: He called her last time we were at the park together
Hannah: I don't know him well
Amanda: Don't be shy, he's very nice
Hannah: If you say so..
Hannah: I'd rather you texted him
Amanda: Just text him 😉
Hannah: Urgh.. Alright
Hannah: Bye
Amanda: Bye bye
"""
    
    # 测试用例 1：检查摘要是否生成并且类型为字符串
    print("Testing case [1/1] started.")
    summary = summarize_conversation(conversation)
    assert isinstance(summary, str), f"Test case [1/1] failed: The summary should be a string, but received type {type(summary)}"
    print("Testing finished.")

# 运行测试函数
test_summarize_conversation()