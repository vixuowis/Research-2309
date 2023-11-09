from transformers import pipeline

def correct_grammar(raw_text: str) -> str:
    """
    This function corrects the grammar of the input text using a pre-trained model from Hugging Face Transformers.

    Args:
        raw_text (str): The text to be corrected.

    Returns:
        str: The corrected text.
    """
    corrector = pipeline('text2text-generation', 'pszemraj/flan-t5-large-grammar-synthesis')
    results = corrector(raw_text)
    return results[0]['generated_text']