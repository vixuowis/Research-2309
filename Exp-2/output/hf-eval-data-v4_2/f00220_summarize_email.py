# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_email(email_text):
    """
    Summarize the content of a long email using PEGASUS model.

    Args:
        email_text (str): The text of the email to be summarized.

    Returns:
        str: The summarized version of the email.

    Raises:
        ValueError: If the email_text is empty.
    """
    if not email_text:
        raise ValueError('The email text must not be empty.')

    # Load the PEGASUS model and tokenizer
    model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_summarizer')
    tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_summarizer')

    # Prepare the text for summarization
    input_ids = tokenizer(email_text, return_tensors='pt').input_ids
    # Generate summary
    summary_ids = model.generate(input_ids)
    # Decode the summary
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary_text

# test_function_code --------------------

def test_summarize_email():
    print("Testing started.")
    # Example of email content
    email_content = "This is a test email content to demonstrate the functionality of the summarize_email function. The function should be able to produce a concise summary of this content."

    # Testing case 1: Valid email content
    print("Testing case [1/1] started.")
    summarized_content = summarize_email(email_content)
    assert summarized_content, "Test case [1/1] failed: Summarization did not return any content."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_email()