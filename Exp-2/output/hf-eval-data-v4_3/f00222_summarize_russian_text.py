# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# function_code --------------------

def summarize_russian_text(text, max_length=1000, num_beams=3, do_sample=False, repetition_penalty=10.0, **kwargs):
    """
    Summarizes a Russian text using a pre-trained 'rut5-base-absum' model.

    Args:
        text (str): The Russian text to be summarized.
        max_length (int): The maximum length of the summary output. Default is 1000.
        num_beams (int): The number of beams for beam search. Default is 3.
        do_sample (bool): If set to True, uses sampling instead of greedy decoding. Default is False.
        repetition_penalty (float): The penalty for repetition. Default is 10.0.
        **kwargs: Additional keyword arguments that will be passed to the generate method of the model.

    Returns:
        str: The summarized text.

    Raises:
        RuntimeError: If the model fails to generate the summary.
    """
    MODEL_NAME = 'cointegrated/rut5-base-absum'
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model.to('cuda')
    model.eval()

    encoded_input = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    with torch.inference_mode():
        summary_ids = model.generate(encoded_input, max_length=max_length, num_beams=num_beams, do_sample=do_sample, repetition_penalty=repetition_penalty, **kwargs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# test_function_code --------------------

def test_summarize_russian_text():
    print("Testing started.")
    # Test case 1: Summarize a short Russian text
    print("Testing case [1/3] started.")
    short_text = 'Пример короткого русского текста.'
    short_summary = summarize_russian_text(short_text)
    assert short_summary != '', 'Test case [1/3] failed: The summary should not be empty.'

    # Test case 2: Summarize a long Russian text
    print("Testing case [2/3] started.")
    long_text = 'Это очень длинный текст на русском языке, который необходимо сократить до короткого резюме, используя предварительно обученную модель.'
    long_summary = summarize_russian_text(long_text)
    assert long_summary != long_text, 'Test case [2/3] failed: The summary should be shorter than the original text.'

    # Test case 3: Summarize with specific parameters
    print("Testing case [3/3] started.")
    param_text = 'Текст для суммирования с использованием определенных параметров.'
    param_summary = summarize_russian_text(param_text, max_length=30, num_beams=5)
    assert len(param_summary) <= 30, 'Test case [3/3] failed: The summary length should not exceed the max_length parameter.'
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_russian_text()