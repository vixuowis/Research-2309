# function_import --------------------

from transformers import BertTokenizerFast, EncoderDecoderModel

# function_code --------------------

def summarize_korean_text(input_text):
    """
    This function takes in a Korean text and returns a summarized version of it.
    
    Args:
        input_text (str): The Korean text to be summarized.
    
    Returns:
        str: The summarized version of the input text.
    
    Raises:
        Exception: If the input is not a string.
    """
    if not isinstance(input_text, str):
        raise Exception('Input must be a string')
    
    tokenizer = BertTokenizerFast.from_pretrained('kykim/bertshared-kor-base')
    model = EncoderDecoderModel.from_pretrained('kykim/bertshared-kor-base')
    
    input_tokens = tokenizer.encode(input_text, return_tensors='pt')
    summary_tokens = model.generate(input_tokens)
    
    return tokenizer.decode(summary_tokens[0], skip_special_tokens=True)

# test_function_code --------------------

def test_summarize_korean_text():
    """
    This function tests the summarize_korean_text function.
    It uses a sample Korean text and checks if the output is a string.
    """
    sample_text = '고객이 입력한 한국어 텍스트를 요약으로 변환하려고 합니다.'
    
    summary = summarize_korean_text(sample_text)
    
    assert isinstance(summary, str), 'Output must be a string'

# call_test_function_code --------------------

test_summarize_korean_text()