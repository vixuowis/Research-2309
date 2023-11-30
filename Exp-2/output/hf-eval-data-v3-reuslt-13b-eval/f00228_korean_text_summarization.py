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
    
    tokenizer = BertTokenizerFast.from_pretrained('kykim/bertshared-kor-base')
    model = EncoderDecoderModel.from_pretrained('kykim/bertshared-kor-base')
    
    tokens = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)

    summary_ids = model.generate(**tokens, 
                                 num_beams=4, 
                                 length_penalty=0.6, 
                                 early_stopping=True, 
                                 min_length=32, 
                                 max_length=512)
    output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
    
    return output


# test_function_code --------------------

def test_korean_text_summarization():
    assert isinstance(korean_text_summarization('고객이 입력한 한국어 텍스트를 요약으로 변환하려고 합니다.'), str)
    assert isinstance(korean_text_summarization('이것은 테스트 문장입니다. 이 문장은 요약되어야 합니다.'), str)
    assert isinstance(korean_text_summarization('한국어 텍스트 요약 알고리즘을 테스트합니다.'), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_korean_text_summarization()