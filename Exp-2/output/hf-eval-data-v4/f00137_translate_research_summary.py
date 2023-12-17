# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5Tokenizer, T5Model

# function_code --------------------

def translate_research_summary(summary_text: str) -> str:
    """
    Translate a research summary from English to Chinese using a pre-trained T5 model.

    Parameters:
    summary_text (str): The research summary in English that needs to be translated.

    Returns:
    str: The translated summary in Chinese.
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5Model.from_pretrained('t5-small', use_cdn=False)
    input_text = f"translate English to Chinese: {summary_text}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    output_ids = model.generate(input_ids)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated_text

# test_function_code --------------------

def test_translate_research_summary():
    print("Testing translation function.")

    # Test case 1: Empty string
    print("Testing case [1/3]: Empty string")
    assert translate_research_summary('') == '', "Test case [1/3] failed: Should return an empty string."

    # Test case 2: Known translation
    print("Testing case [2/3]: Known translation")
    test_summary = "Climate change is a global challenge."
    expected_translation = "气候变化是一个全球性的挑战。" # This might not be accurate and it's for the sake of the example
    assert translate_research_summary(test_summary) == expected_translation, "Test case [2/3] failed: The translation is incorrect."

    # Test case 3: Non-English input
    print("Testing case [3/3]: Non-English input")
    non_english_summary = "这是中文摘要。"
    assert translate_research_summary(non_english_summary) == non_english_summary, "Test case [3/3] failed: Should return the original non-English text."
    print("Testing completed successfully.")

# Run the test function
test_translate_research_summary()