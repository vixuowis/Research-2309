from transformers import pipeline


def fill_in_the_blanks(sentence: str) -> str:
    """
    This function uses the BERT large model (uncased) to fill in the blanks in a sentence.

    Args:
        sentence (str): The sentence with a '[MASK]' token representing the missing word.

    Returns:
        str: The sentence with the '[MASK]' token replaced by the predicted word.
    """
    fill_in_the_blanks = pipeline('fill-mask', model='bert-large-uncased')
    filled_sentence = fill_in_the_blanks(sentence)
    return filled_sentence[0]['sequence']