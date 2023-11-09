# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def summarize_french_article(input_text: str) -> str:
    """
    Summarizes a French article using the T5ForConditionalGeneration model from the Hugging Face Transformers library.

    Args:
        input_text (str): The French article to be summarized.

    Returns:
        str: The summarized text.
    """
    tokenizer = T5Tokenizer.from_pretrained('plguillou/t5-base-fr-sum-cnndm')
    model = T5ForConditionalGeneration.from_pretrained('plguillou/t5-base-fr-sum-cnndm')
    input_tokens = tokenizer.encode('summarize: ' + input_text, return_tensors='pt')
    summary_ids = model.generate(input_tokens)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# test_function_code --------------------

def test_summarize_french_article():
    """
    Tests the summarize_french_article function by summarizing a sample French article.
    """
    input_text = 'Selon un rapport récent, les constructeurs automobiles prévoient...'
    summary = summarize_french_article(input_text)
    assert isinstance(summary, str) and len(summary) > 0

# call_test_function_code --------------------

test_summarize_french_article()