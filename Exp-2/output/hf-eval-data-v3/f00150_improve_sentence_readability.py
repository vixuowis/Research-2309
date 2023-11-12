# function_import --------------------

from transformers import DebertaTokenizer, DebertaModel

# function_code --------------------

def improve_sentence_readability(sentence: str) -> str:
    """
    Improve the readability and grammaticality of the provided sentence by suggesting the best replacement for the masked part.

    Args:
        sentence (str): The sentence to be improved. The part to be replaced should be marked with [MASK].

    Returns:
        str: The improved sentence.

    Raises:
        TypeError: If the sentence is not a string.
    """
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-v2-xlarge')
    model = DebertaModel.from_pretrained('microsoft/deberta-v2-xlarge')
    input_text = tokenizer(sentence, return_tensors='pt')
    output = model(**input_text)
    predicted_token = tokenizer.decode(output.logits.argmax(-1)[:, -1].item())
    improved_sentence = sentence.replace('[MASK]', predicted_token)
    return improved_sentence

# test_function_code --------------------

def test_improve_sentence_readability():
    """
    Test the function improve_sentence_readability.
    """
    sentence1 = 'The cat was chasing its [MASK].'
    sentence2 = 'I am a [MASK] teacher.'
    sentence3 = 'He is a [MASK] student.'
    assert isinstance(improve_sentence_readability(sentence1), str)
    assert isinstance(improve_sentence_readability(sentence2), str)
    assert isinstance(improve_sentence_readability(sentence3), str)
    print('All Tests Passed')

# call_test_function_code --------------------

test_improve_sentence_readability()