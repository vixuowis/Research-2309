# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import BertTokenizerFast, EncoderDecoderModel

# function_code --------------------

def summarize_korean_text(input_text):
    """
    Summarizes the given Korean text using a pre-trained BERT model.

    Args:
        input_text (str): The Korean text to be summarized.

    Returns:
        str: The summarized version of the input text.

    Raises:
        ValueError: If the input text is not a valid string.
    """
    if not isinstance(input_text, str):
        raise ValueError('Input text must be a valid string.')

    tokenizer = BertTokenizerFast.from_pretrained('kykim/bertshared-kor-base')
    model = EncoderDecoderModel.from_pretrained('kykim/bertshared-kor-base')
    input_tokens = tokenizer.encode(input_text, return_tensors='pt')
    summary_tokens = model.generate(input_tokens)
    summary_text = tokenizer.decode(summary_tokens[0], skip_special_tokens=True)
    return summary_text

# test_function_code --------------------

def test_summarize_korean_text():
    print("Testing started.")

    # Testing case 1: Normal text
    print("Testing case [1/1] started.")
    sample_text = "고객이 입력한 한국어 텍스트를 요약으로 변환하려고 합니다."
    summary = summarize_korean_text(sample_text)
    assert isinstance(summary, str), f"Test case [1/1] failed: Expected string but got {type(summary)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_korean_text()