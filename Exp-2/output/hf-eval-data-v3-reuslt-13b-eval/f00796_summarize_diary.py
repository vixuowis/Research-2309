# function_import --------------------

from transformers import LEDForConditionalGeneration, AutoTokenizer

# function_code --------------------

def summarize_diary(diary_entry: str) -> str:
    '''
    Summarizes a given diary entry using the pre-trained model 'MingZhong/DialogLED-base-16384'.

    Args:
        diary_entry (str): The diary entry to be summarized.

    Returns:
        str: The summarized text.
    '''
    tokenizer = AutoTokenizer.from_pretrained('MingZhong/DialogLED-base-16384')
    model = LEDForConditionalGeneration.from_pretrained('MingZhong/DialogLED-base-16384').to('cuda')
    
    batch = tokenizer([diary_entry], return_tensors='pt', truncation=True, padding="max_length",  max_length=2000).to('cuda')
    summary_ids = model.generate(batch['input_ids'], num_beams=4, max_length=512, min_length=55)
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)


# test_function_code --------------------

def test_summarize_diary():
    '''
    Tests the function summarize_diary.
    '''
    diary_entry1 = 'Today was a great day. I managed to fix the issue with the oxygen generator and had a successful communication session with the ground control.'
    diary_entry2 = 'I had a tough day today. The solar panels were not working properly and I had to spend the whole day fixing them.'
    diary_entry3 = 'Today was a normal day. I did my routine checks and everything seems to be working fine.'

    assert len(summarize_diary(diary_entry1)) > 0
    assert len(summarize_diary(diary_entry2)) > 0
    assert len(summarize_diary(diary_entry3)) > 0

    return 'All Tests Passed'


# call_test_function_code --------------------

test_summarize_diary()