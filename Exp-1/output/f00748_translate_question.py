from typing import *
from translator import translator

def translate_question(question: str, src_lang: str, tgt_lang: str) -> str:
    """Translate the given question from the source language to the target language.

    Args:
        question (str): The question to be translated.
        src_lang (str): The source language of the question.
        tgt_lang (str): The target language to translate the question to.

    Returns:
        str: The translated question."""
    translated_question = translator(question=question, src_lang=src_lang, tgt_lang=tgt_lang)
    return translated_question
