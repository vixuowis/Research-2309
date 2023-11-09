# function_import --------------------

from transformers import AutoModelWithLMHead, AutoTokenizer

# function_code --------------------

def gen_sentence(words, max_length=32):
    """
    Generate a creative sentence using the provided words.

    Args:
        words (str): A string of words to be included in the sentence.
        max_length (int, optional): The maximum length of the generated sentence. Defaults to 32.

    Returns:
        str: A creative sentence including the provided words.
    """
    tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-common_gen')
    model = AutoModelWithLMHead.from_pretrained('mrm8488/t5-base-finetuned-common_gen')
    input_text = words
    features = tokenizer([input_text], return_tensors='pt')
    output = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'], max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# test_function_code --------------------

def test_gen_sentence():
    """
    Test the function gen_sentence.
    """
    words = 'moon rabbit forest magic'
    sentence = gen_sentence(words)
    assert isinstance(sentence, str)
    assert all(word in sentence for word in words.split())

# call_test_function_code --------------------

test_gen_sentence()