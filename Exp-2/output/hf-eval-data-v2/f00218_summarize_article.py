# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def summarize_article(article_text: str, model_name: str = 'csebuetnlp/mT5_multilingual_XLSum') -> str:
    """
    Summarizes a given news article using a pre-trained model.

    Args:
        article_text (str): The text of the article to be summarized.
        model_name (str, optional): The name of the pre-trained model to use. Defaults to 'csebuetnlp/mT5_multilingual_XLSum'.

    Returns:
        str: The summarized text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(article_text, return_tensors='pt', padding=True, truncation=True, max_length=512).input_ids
    output_ids = model.generate(input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)[0]
    summary = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return summary

# test_function_code --------------------

def test_summarize_article():
    """
    Tests the summarize_article function.
    """
    article_text = 'Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said. The policy includes the termination of accounts of anti-vaccine influencers. Tech giants have been criticised for not doing more to counter false health information on their sites.'
    summary = summarize_article(article_text)
    assert isinstance(summary, str) and len(summary) > 0, 'The function should return a non-empty string.'

# call_test_function_code --------------------

test_summarize_article()