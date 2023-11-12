# function_import --------------------

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# function_code --------------------

MODEL_NAME = 'cointegrated/rut5-base-absum'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model.eval()

def summarize_russian_text(text, max_length=1000, num_beams=3, do_sample=False, repetition_penalty=10.0, **kwargs):
    """
    Summarize the given Russian text.

    Args:
        text (str): The Russian text to be summarized.
        max_length (int, optional): The maximum length of the summary. Defaults to 1000.
        num_beams (int, optional): The number of beams for beam search. Defaults to 3.
        do_sample (bool, optional): Whether to do sampling. Defaults to False.
        repetition_penalty (float, optional): The penalty for repetition. Defaults to 10.0.
        **kwargs: Additional parameters for the 'generate' function of the model.

    Returns:
        str: The summary of the input text.
    """
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(x, max_length=max_length, num_beams=num_beams, do_sample=do_sample, repetition_penalty=repetition_penalty, **kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# test_function_code --------------------

def test_summarize_russian_text():
    """
    Test the 'summarize_russian_text' function.
    """
    russian_text = 'Пример оригинального русского текста здесь...'
    summary = summarize_russian_text(russian_text)
    assert isinstance(summary, str)
    assert len(summary) <= 1000

    russian_text = 'Это очень длинный русский текст, который нужно сократить до короткого резюме...'
    summary = summarize_russian_text(russian_text, max_length=500)
    assert isinstance(summary, str)
    assert len(summary) <= 500

    russian_text = 'Это еще один русский текст, который нужно сократить...'
    summary = summarize_russian_text(russian_text, num_beams=5)
    assert isinstance(summary, str)
    assert len(summary) <= 1000

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_summarize_russian_text())