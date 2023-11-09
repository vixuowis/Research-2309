from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def translate_english_to_german(src_text):
    """
    Translates English text to German using the MBartForConditionalGeneration model.

    Args:
        src_text (str): The source text in English to be translated.

    Returns:
        str: The translated text in German.
    """
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
    tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50', src_lang='en_XX', tgt_lang='de_DE')
    translated_output = model.generate(**tokenizer(src_text, return_tensors='pt'))
    tgt_text = tokenizer.batch_decode(translated_output, skip_special_tokens=True)
    return tgt_text[0]