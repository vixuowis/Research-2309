# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import LEDForConditionalGeneration, AutoTokenizer

# function_code --------------------

def summarize_diary_entry(diary_entry):
    """
    Summarize a given diary entry using the DialogLED-base-16384 pre-trained model.

    Parameters:
    diary_entry (str): Text of the diary entry to be summarized.

    Returns:
    str: The summarized text of the diary entry.
    """
    # Load the pre-trained model and tokenizer
    model = LEDForConditionalGeneration.from_pretrained('MingZhong/DialogLED-base-16384')
    tokenizer = AutoTokenizer.from_pretrained('MingZhong/DialogLED-base-16384')

    # Tokenize the diary entry
    input_tokens = tokenizer(diary_entry, return_tensors='pt')

    # Generate the summary output tokens
    summary_output = model.generate(**input_tokens)

    # Decode the summary tokens into text
    summary_text = tokenizer.decode(summary_output[0], skip_special_tokens=True)
    return summary_text

# test_function_code --------------------

def test_summarize_diary_entry():
    print("Testing summarize_diary_entry function.")
    # Example entries to test summarization
    entries_to_summarize = [
        ('Entry 1: Today was an amazing day at the International Space Station...', 'Today was amazing at the ISS...'),
        ('Entry 2: Completed several experiments on plant growth in zero gravity...', 'Completed experiments on plant growth...')
    ]

    for idx, (entry, expected_summary) in enumerate(entries_to_summarize):
        print(f"Testing case [{idx + 1}/{len(entries_to_summarize)}] started.")
        summary = summarize_diary_entry(entry)
        assert summary.startswith(expected_summary), f"Test case [{idx + 1}/{len(entries_to_summarize)}] failed: Expected summary to start with '{expected_summary}', but got '{summary}'."
        print(f"Testing case [{idx + 1}/{len(entries_to_summarize)}] succeed.")
    print("All tests passed!")

# Run the test function
test_summarize_diary_entry()