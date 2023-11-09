from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))


def summarize_text(article_text: str, model_name: str = 'csebuetnlp/mT5_multilingual_XLSum') -> str:
    """
    Summarizes the given article text using the specified model.

    Args:
        article_text (str): The text of the article to be summarized.
        model_name (str, optional): The name of the model to be used for summarization. Defaults to 'csebuetnlp/mT5_multilingual_XLSum'.

    Returns:
        str: The summarized text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    input_ids = tokenizer(
        [WHITESPACE_HANDLER(article_text)],
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=512
    )['input_ids']
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]
    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return summary