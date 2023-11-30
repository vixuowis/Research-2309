# function_import --------------------

from transformers import BertTokenizerFast, EncoderDecoderModel

# function_code --------------------

def korean_text_summarization(input_text: str) -> str:
    """
    Summarize the input Korean text using the pretrained model 'kykim/bertshared-kor-base'.

    Args:
        input_text (str): The input Korean text to be summarized.

    Returns:
        str: The summarized text.
    """

    # Load tokenizer and encoder-decoder model --------------------
    fast_tokenizer = BertTokenizerFast.from_pretrained('kykim/bertshared-kor-base')
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('kykim/bertshared-kor-base', 'kykim/bertshared-kor-base')

    # Prepare input_text --------------------
    tokenized_input_text = fast_tokenizer(input_text, return_tensors='pt')['input_ids']

    # Summarize input_text --------------------
    summarized_text = model.generate(**tokenized_input_text)

    # Decode summarized text and return it --------------------
    return fast_tokenizer.decode(summarized_text[0], skip_special_tokens=True)


# test_function_code --------------------

def test_korean_text_summarization():
    assert isinstance(korean_text_summarization('고객이 입력한 한국어 텍스트를 요약으로 변환하려고 합니다.'), str)
    assert isinstance(korean_text_summarization('이것은 테스트 문장입니다. 이 문장은 요약되어야 합니다.'), str)
    assert isinstance(korean_text_summarization('한국어 텍스트 요약 알고리즘을 테스트합니다.'), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_korean_text_summarization()