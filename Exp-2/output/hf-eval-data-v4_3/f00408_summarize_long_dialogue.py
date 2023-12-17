# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import LEDForConditionalGeneration, LEDTokenizer

# function_code --------------------

def summarize_long_dialogue(input_text):
    """
    Summarizes a long dialogue using the DialogLED model.

    Args:
        input_text (str): The dialogue text to be summarized.

    Returns:
        str: The summarized version of the dialogue.

    Raises:
        ValueError: If `input_text` is not provided.
    """
    if not input_text:
        raise ValueError('No input text provided for summarization.')

    model = LEDForConditionalGeneration.from_pretrained('MingZhong/DialogLED-base-16384')
    tokenizer = LEDTokenizer.from_pretrained('MingZhong/DialogLED-base-16384')

    input_tokens = tokenizer.encode(input_text, return_tensors='pt')
    summary_ids = model.generate(input_tokens)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# test_function_code --------------------

def test_summarize_long_dialogue():
    print("Testing started.")
    sample_data = 'This is a very long and complex dialogue text that spans over multiple topics and requires summarization.'

    # Test case 1: Valid summary generation
    print("Testing case [1/1] started.")
    summary = summarize_long_dialogue(sample_data)
    assert len(summary) < len(sample_data), f"Test case [1/1] failed: Summary not shorter than input."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_long_dialogue()