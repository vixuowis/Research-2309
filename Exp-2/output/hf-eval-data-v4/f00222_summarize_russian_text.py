# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# function_code --------------------

def summarize_russian_text(text, max_length=1000, num_beams=3, do_sample=False, repetition_penalty=10.0, **kwargs):
    MODEL_NAME = 'cointegrated/rut5-base-absum'
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model.cuda()
    model.eval()

    input_text = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    with torch.inference_mode():
        summary_ids = model.generate(input_text, max_length=max_length, num_beams=num_beams, do_sample=do_sample, repetition_penalty=repetition_penalty, **kwargs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_russian_text():
    print("Testing summarize_russian_text function.")
    russian_text = "Пример текста на русском языке для тестирования функции."
    summary = summarize_russian_text(russian_text)
    print("Summary: ", summary)
    assert type(summary) == str, "The function should return a string."
    assert summary != russian_text, "The summary should not be the same as the input text."
    print("All tests passed!")

test_summarize_russian_text()