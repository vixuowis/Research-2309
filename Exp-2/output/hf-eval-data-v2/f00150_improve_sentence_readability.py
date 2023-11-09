# function_import --------------------

from transformers import DebertaTokenizer, DebertaModel

# function_code --------------------

def improve_sentence_readability(sentence: str) -> str:
    """
    Improve the readability and grammaticality of the provided sentence by suggesting the best replacement for the masked part.

    Args:
        sentence (str): The sentence with a masked part ([MASK]) to be replaced.

    Returns:
        str: The improved sentence with the masked part replaced by the best suggestion.
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
    sentence = 'The cat was chasing its [MASK].'
    improved_sentence = improve_sentence_readability(sentence)
    assert '[MASK]' not in improved_sentence, 'The function did not replace the masked part.'

# call_test_function_code --------------------

test_improve_sentence_readability()