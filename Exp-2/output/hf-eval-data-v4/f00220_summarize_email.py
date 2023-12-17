# requirements_file --------------------

!pip install -U transformers torch sentencepiece

# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_email(email_text):
    """
    Summarizes the contents of a long email using the PEGASUS model.

    Parameters:
        email_text (str): The text of the email to summarize.

    Returns:
        str: A summarized version of the email.
    """
    model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_summarizer')
    tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_summarizer')

    input_ids = tokenizer(email_text, return_tensors='pt').input_ids
    summary_ids = model.generate(input_ids)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# test_function_code --------------------

def test_summarize_email():
    print("Testing summarize_email function.")

    # Test case: Summarizing a generic email text
    email_text = "Dear team, as you are aware, we are facing certain challenges in the current project due to unexpected market conditions..."
    summary = summarize_email(email_text)
    assert len(summary) < len(email_text), f"Test case failed: The summary is not shorter than the original email text."

    print("Test passed. The email summary is shorter than the original text.")

# Run the test function
test_summarize_email()