# function_import --------------------

from transformers import T5Tokenizer, T5Model

# function_code --------------------

def translate_research_summary(research_summary):
    """
    This function translates a given research summary from English to Chinese using the T5 small model from Hugging Face Transformers.

    Args:
        research_summary (str): The research summary that needs to be translated from English to Chinese.

    Returns:
        str: The translated research summary in Chinese.
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5Model.from_pretrained('t5-small')
    input_text = f"translate English to Chinese: {research_summary}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    decoded_text = model.generate(input_ids)
    translated_summary = tokenizer.batch_decode(decoded_text, skip_special_tokens=True)
    return translated_summary[0]

# test_function_code --------------------

def test_translate_research_summary():
    """
    This function tests the translate_research_summary function by using a sample research summary and checking if the output is not None.
    """
    research_summary = 'Climate change is a significant and lasting change in the statistical distribution of weather patterns over periods ranging from decades to millions of years.'
    translated_summary = translate_research_summary(research_summary)
    assert translated_summary is not None, 'The translation function did not return a result.'

# call_test_function_code --------------------

test_translate_research_summary()