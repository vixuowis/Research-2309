from transformers import LEDForConditionalGeneration, AutoTokenizer


def summarize_diary(diary_entry: str) -> str:
    """
    Summarizes a given diary entry using the pre-trained model 'MingZhong/DialogLED-base-16384'.

    Args:
        diary_entry (str): The diary entry to be summarized.

    Returns:
        str: The summarized text.
    """
    model = LEDForConditionalGeneration.from_pretrained('MingZhong/DialogLED-base-16384')
    tokenizer = AutoTokenizer.from_pretrained('MingZhong/DialogLED-base-16384')

    input_tokens = tokenizer(diary_entry, return_tensors='pt')
    summary_output = model.generate(**input_tokens)
    summary_text = tokenizer.decode(summary_output[0])

    return summary_text