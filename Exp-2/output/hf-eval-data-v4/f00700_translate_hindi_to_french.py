# requirements_file --------------------

!pip install -U transformers==4.0.0

# function_import --------------------

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# function_code --------------------

def translate_hindi_to_french(hindi_text: str) -> str:
    """
    Translates a given message from Hindi to French using the MBart large model.

    :param hindi_text: str, The input text in Hindi language.
    :return: str, The translated text in French language.
    """
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    tokenizer.src_lang = 'hi_IN'

    inputs = tokenizer(hindi_text, return_tensors='pt')
    forced_bos_token_id = tokenizer.lang_code_to_id['fr_XX']
    generated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

# test_function_code --------------------

def test_translate_hindi_to_french():
    hindi_text = 'आपकी प्रेज़टेशन का आधार अच्छा था, लेकिन डेटा विश्लेषण पर ध्यान देना चाहिए।'
    expected_translation_start = 'La base de votre pr'
    print('Testing translate_hindi_to_french function.')
    
    translation = translate_hindi_to_french(hindi_text)
    assert translation.startswith(expected_translation_start), f'Translation did not start with expected French phrase. Got: {translation}'
    print('Translation matches expected start. Test success.')

    print('Testing finished.')

# Running the test function
test_translate_hindi_to_french()