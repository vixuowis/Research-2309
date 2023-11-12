# function_import --------------------

from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM

# function_code --------------------

def generate_fill_in_the_blank_question(sentence: str, mask_index: int) -> str:
    '''
    Generate a fill-in-the-blank question from a given sentence by masking a word at a specific index.

    Args:
        sentence (str): The sentence to generate the question from.
        mask_index (int): The index of the word to mask.

    Returns:
        str: The sentence with the masked word.
    '''
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xxlarge')
    model = DebertaV2ForMaskedLM.from_pretrained('microsoft/deberta-v2-xxlarge')
    words = sentence.split()
    words[mask_index] = '[MASK]'
    masked_sentence = ' '.join(words)
    inputs = tokenizer(masked_sentence, return_tensors='pt')
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1)
    masked_word = tokenizer.decode(predictions[0][mask_index])
    new_sentence = masked_sentence.replace('[MASK]', masked_word)
    return new_sentence

# test_function_code --------------------

def test_generate_fill_in_the_blank_question():
    '''
    Test the function generate_fill_in_the_blank_question.
    '''
    sentence = 'The cat chased the mouse and then climbed the tree.'
    assert generate_fill_in_the_blank_question(sentence, 3) == 'The cat chased the [MASK] and then climbed the tree.'
    assert generate_fill_in_the_blank_question(sentence, 0) == '[MASK] cat chased the mouse and then climbed the tree.'
    assert generate_fill_in_the_blank_question(sentence, 6) == 'The cat chased the mouse and then [MASK] the tree.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_fill_in_the_blank_question()