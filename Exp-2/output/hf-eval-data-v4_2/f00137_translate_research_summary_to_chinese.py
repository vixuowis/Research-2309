# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def translate_research_summary_to_chinese(research_summary: str) -> str:
    """
    Translates an English research summary into Chinese using the T5-small model.

    Args:
        research_summary (str): The research summary in English to be translated.

    Returns:
        str: The translated research summary in Chinese.

    Raises:
        ValueError: If the input research summary is not a string or is empty.
    """
    if not isinstance(research_summary, str) or not research_summary:
        raise ValueError('Input research summary must be a non-empty string.')

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    input_text = f'translate English to Chinese: {research_summary}'
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    translated_summary_ids = model.generate(input_ids)
    translated_summary = tokenizer.decode(translated_summary_ids[0], skip_special_tokens=True)
    return translated_summary


# test_function_code --------------------

from datasets import load_dataset

english_summary = 'Climate change contributes to extreme weather events.'
expected_translation = '...'  # Expected translation goes here

# The load_dataset functionality is a placeholder and not actually required for this function.

# call_test_function_line --------------------

test_translate_research_summary_to_chinese()