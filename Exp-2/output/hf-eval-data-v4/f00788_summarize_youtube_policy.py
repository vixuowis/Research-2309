# requirements_file --------------------

!pip install -U transformers==4.11.0.dev0

# function_import --------------------

import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def summarize_youtube_policy(article_text, model_name):
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

def test_summarize_youtube_policy():
    print("Testing summarize_youtube_policy function.")
    article_text = "Videos that say approved vaccines are dangerous and cause autism, cancer or infertility..."
    model_name = 'csebuetnlp/mT5_multilingual_XLSum'
    summary = summarize_youtube_policy(article_text, model_name)
    expected_summary = "YouTube will remove videos spreading misinformation about approved vaccines..."
    assert summary == expected_summary, f"Test failed: expected {expected_summary}, got {summary}"
    print("Test passed.")

# Run the test function
test_summarize_youtube_policy()