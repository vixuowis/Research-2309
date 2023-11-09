# function_import --------------------

from transformers import LEDForConditionalGeneration, LEDTokenizer

# function_code --------------------

def summarize_dialogue(input_text):
    """
    Summarizes a given dialogue using the pre-trained DialogLED model.

    Args:
        input_text (str): The dialogue to be summarized.

    Returns:
        str: The summarized dialogue.
    """
    model = LEDForConditionalGeneration.from_pretrained('MingZhong/DialogLED-base-16384')
    tokenizer = LEDTokenizer.from_pretrained('MingZhong/DialogLED-base-16384')
    input_tokens = tokenizer.encode(input_text, return_tensors='pt')
    summary_ids = model.generate(input_tokens)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_dialogue():
    """
    Tests the summarize_dialogue function by summarizing a sample dialogue.
    """
    sample_dialogue = 'Hello, how are you? I am fine. That is good to hear.'
    summary = summarize_dialogue(sample_dialogue)
    assert isinstance(summary, str), 'The output should be a string.'
    assert len(summary) < len(sample_dialogue), 'The summary should be shorter than the original dialogue.'

# call_test_function_code --------------------

test_summarize_dialogue()