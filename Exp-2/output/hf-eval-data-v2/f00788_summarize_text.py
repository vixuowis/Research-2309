# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# function_code --------------------

def summarize_text(article_text: str, model_name: str = 'csebuetnlp/mT5_multilingual_XLSum') -> str:
    """
    Summarizes the given article text using the specified model.

    Args:
        article_text (str): The text of the article to be summarized.
        model_name (str, optional): The name of the model to be used for summarization. Defaults to 'csebuetnlp/mT5_multilingual_XLSum'.

    Returns:
        str: The summarized text.
    """
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
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

# test_function_code --------------------

def test_summarize_text():
    """
    Tests the summarize_text function.
    """
    article_text = 'Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said. The policy includes the termination of accounts of anti-vaccine influencers. Tech giants have been criticised for not doing more to counter false health information on their sites. In July, US President Joe Biden said social media platforms were largely responsible for people's scepticism in getting vaccinated by spreading misinformation, and appealed for them to address the issue. YouTube, which is owned by Google, said 130,000 videos were removed from its platform since last year, when it implemented a ban on content spreading misinformation about Covid vaccines. In a blog post, the company said it had seen false claims about Covid jabs spill over into misinformation about vaccines in general. The new policy covers long-approved vaccines, such as those against measles or hepatitis B. We're expanding our medical misinformation policies on YouTube with new guidelines on currently administered vaccines that are approved and confirmed to be safe and effective by local health authorities and the WHO, the post said, referring to the World Health Organization.'
    summary = summarize_text(article_text)
    assert isinstance(summary, str), 'The result should be a string.'
    assert len(summary) > 0, 'The summary should not be empty.'

# call_test_function_code --------------------

test_summarize_text()