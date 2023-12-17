# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import BertTokenizerFast, EncoderDecoderModel

# function_code --------------------

def summarize_korean_text(input_text):
    """
    Summarizes the input Korean text using a pretrained BERT-based Encoder-Decoder model.

    Args:
        input_text (str): A string containing the Korean text to be summarized.

    Returns:
        str: The summarized version of the input text.
    """
    tokenizer = BertTokenizerFast.from_pretrained('kykim/bertshared-kor-base')
    model = EncoderDecoderModel.from_pretrained('kykim/bertshared-kor-base')
    input_tokens = tokenizer(input_text, return_tensors='pt')
    summary_tokens = model.generate(input_tokens['input_ids'], attention_mask=input_tokens['attention_mask'])
    summary = tokenizer.decode(summary_tokens[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_korean_text():
    print("Testing the summarize_korean_text function.")
    sample_text = "고객이 입력한 한국어 텍스트를 요약으로 변환하려고 합니다."

    # Test case 1: Sample text summarization
    print("Testing case [1/1] started.")
    summary = summarize_korean_text(sample_text)
    assert len(summary) < len(sample_text), f"Test case [1/1] failed: The summary should be shorter than the input text."
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_summarize_korean_text()