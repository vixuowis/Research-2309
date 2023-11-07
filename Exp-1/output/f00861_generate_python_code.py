from typing import *
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def generate_python_code(src_lang, tgt_lang, input_code):
    '''
    Translate code from the source language to the target language using the Facebook NLLB-200-Distilled-600M model.

    Args:
        src_lang (str): The BCP-47 code of the source language.
        tgt_lang (str): The BCP-47 code of the target language.
        input_code (str): The code to be translated.

    Returns:
        str: The translated code.
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        'facebook/nllb-200-distilled-600M', use_auth_token=True, src_lang=src_lang
    )
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M', use_auth_token=True)

    inputs = tokenizer(input_code, return_tensors='pt')

    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang], max_length=100
    )

    translated_code = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    return translated_code
