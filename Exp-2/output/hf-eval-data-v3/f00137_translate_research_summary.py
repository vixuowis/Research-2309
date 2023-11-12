# function_import --------------------

from transformers import T5Tokenizer, T5Model

# function_code --------------------

def translate_research_summary(research_summary):
    """
    Translates a research summary from English to Chinese using the T5 model.

    Args:
        research_summary (str): The research summary in English.

    Returns:
        str: The translated research summary in Chinese.

    Raises:
        ReadTimeout: If the request to the Hugging Face model server times out.
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
    Tests the translate_research_summary function with a few research summaries.
    """
    research_summary1 = "Climate change is a significant and lasting change in the statistical distribution of weather patterns over periods ranging from decades to millions of years."
    research_summary2 = "The main cause of current global warming is human activity, primarily the emission of greenhouse gases."
    research_summary3 = "Climate change can have wide-ranging effects on the environment, the economy, and human health."
    assert isinstance(translate_research_summary(research_summary1), str)
    assert isinstance(translate_research_summary(research_summary2), str)
    assert isinstance(translate_research_summary(research_summary3), str)
    print('All Tests Passed')

# call_test_function_code --------------------

test_translate_research_summary()