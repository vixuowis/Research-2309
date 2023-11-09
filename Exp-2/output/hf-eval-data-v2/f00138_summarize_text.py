# function_import --------------------

from transformers import BartTokenizer, BartForConditionalGeneration

# function_code --------------------

def summarize_text(input_text):
    """
    Summarizes the given text using the pre-trained model 'sshleifer/distilbart-cnn-12-6'.

    Args:
        input_text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
    tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    inputs = tokenizer(input_text, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=50, early_stopping=True)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# test_function_code --------------------

def test_summarize_text():
    """
    Tests the function 'summarize_text'.
    """
    input_text = 'This is a long article text that needs to be summarized.'
    summary = summarize_text(input_text)
    assert isinstance(summary, str), 'The output should be a string.'
    assert len(summary) < len(input_text), 'The summary should be shorter than the input text.'

# call_test_function_code --------------------

test_summarize_text()