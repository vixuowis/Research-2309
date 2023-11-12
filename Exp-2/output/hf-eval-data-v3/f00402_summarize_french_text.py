# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def summarize_french_text(input_text: str) -> str:
    """
    Summarize a given French text using the T5ForConditionalGeneration model from the Hugging Face Transformers library.

    Args:
        input_text (str): The French text to be summarized.

    Returns:
        str: The summarized text.

    Raises:
        ValueError: If the input_text is not a string.
    """
    if not isinstance(input_text, str):
        raise ValueError('Input text must be a string.')

    # Load the pre-trained French summarization model and the corresponding tokenizer
    tokenizer = T5Tokenizer.from_pretrained('plguillou/t5-base-fr-sum-cnndm', legacy=False)
    model = T5ForConditionalGeneration.from_pretrained('plguillou/t5-base-fr-sum-cnndm')

    # Tokenize the input article text
    input_tokens = tokenizer.encode('summarize: ' + input_text, return_tensors='pt')

    # Generate the summary
    summary_ids = model.generate(input_tokens)

    # Decode the generated summary tokens to get the final summary text
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary_text

# test_function_code --------------------

def test_summarize_french_text():
    """Tests for the `summarize_french_text` function"""
    # Test case: Normal case with a French text
    input_text = 'Selon un rapport récent, les constructeurs automobiles prévoient d\'accélérer la production de voitures électriques et de réduire la production de voitures à moteur à combustion interne.'
    summary = summarize_french_text(input_text)
    assert isinstance(summary, str), 'The result is not a string.'

    # Test case: The input is not a string
    try:
        summarize_french_text(123)
    except ValueError as e:
        assert str(e) == 'Input text must be a string.', 'The ValueError message is not correct.'

    # Test case: The input is an empty string
    summary = summarize_french_text('')
    assert summary == '', 'The result is not an empty string for an empty input.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_french_text()