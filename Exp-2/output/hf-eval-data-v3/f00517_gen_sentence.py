# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer

# function_code --------------------

def gen_sentence(words, max_length=32):
    """
    Generate a sentence based on the given words using a pre-trained model.

    Args:
        words (str): A string of words to be used for sentence generation.
        max_length (int, optional): The maximum length of the generated sentence. Defaults to 32.

    Returns:
        str: The generated sentence.
    """
    tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-common_gen')
    model = AutoModelForCausalLM.from_pretrained('mrm8488/t5-base-finetuned-common_gen')
    input_text = words
    features = tokenizer([input_text], return_tensors='pt')
    output = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'], max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# test_function_code --------------------

def test_gen_sentence():
    """
    Test the gen_sentence function.
    """
    test_words = 'tree plant ground hole dig'
    generated_sentence = gen_sentence(test_words)
    assert isinstance(generated_sentence, str), 'The output should be a string.'
    assert len(generated_sentence.split()) <= 32, 'The output sentence should not exceed the maximum length.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_gen_sentence()