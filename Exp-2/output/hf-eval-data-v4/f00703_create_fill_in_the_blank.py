# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from random import randint
from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM
import torch

# function_code --------------------

tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xxlarge')
model = DebertaV2ForMaskedLM.from_pretrained('microsoft/deberta-v2-xxlarge')

def create_fill_in_the_blank(sentence):
    """
    Create a fill-in-the-blank question by masking a random word in the sentence.
    
    :param sentence: Original sentence from which to create quiz.
    :return: A tuple with the fill-in-the-blank question and the masked word.
    """
    tokens = tokenizer.tokenize(sentence)
    word_indices = [i for i, token in enumerate(tokens) if token.isalpha()]
    masked_index = randint(0, len(word_indices) - 1)
    mask_position = word_indices[masked_index]
    tokens[mask_position] = tokenizer.mask_token
    masked_sentence = tokenizer.convert_tokens_to_string(tokens)
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    mask_token_logits = outputs.logits[0, mask_position]
    masked_word_id = mask_token_logits.argmax().item()
    masked_word = tokenizer.decode([masked_word_id])
    return masked_sentence.replace(tokenizer.mask_token, '[MASK]'), masked_word

# test_function_code --------------------

def test_create_fill_in_the_blank():
    print("Testing started.")
    sentence = "The quick brown fox jumps over the lazy dog."
    question, answer = create_fill_in_the_blank(sentence)
    assert '[MASK]' in question and answer, "Test case [1/1] failed: The function did not return a fill-in-the-blank question and an answer"
    print(f"Generated Question: {question}, Answer: {answer}")
    print("Testing finished.")

test_create_fill_in_the_blank()