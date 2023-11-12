# function_import --------------------

from transformers import MarianMTModel, MarianTokenizer

# function_code --------------------

def translate_portuguese_lyrics(src_text):
    '''
    Translate Portuguese lyrics into English using the MarianMT model.

    Args:
        src_text (list): A list of strings where each string is a line of lyrics in Portuguese.

    Returns:
        translated_lyrics (list): A list of strings where each string is a line of translated lyrics in English.
    '''
    model_name = 'Helsinki-NLP/opus-mt-pt-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    batch = tokenizer.prepare_seq2seq_batch(src_text)
    gen = model.generate(**batch)
    translated_lyrics = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return translated_lyrics

# test_function_code --------------------

def test_translate_portuguese_lyrics():
    '''
    Test the function translate_portuguese_lyrics.
    '''
    src_text = ['O sol brilha no céu', 'A lua brilha no mar']
    translated_lyrics = translate_portuguese_lyrics(src_text)
    assert isinstance(translated_lyrics, list), 'The result should be a list.'
    assert len(translated_lyrics) == len(src_text), 'The number of translated lines should be equal to the number of input lines.'
    assert all(isinstance(line, str) for line in translated_lyrics), 'Each line in the result should be a string.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_portuguese_lyrics()