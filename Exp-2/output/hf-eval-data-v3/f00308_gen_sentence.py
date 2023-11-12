# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

# function_code --------------------

def gen_sentence(words: str, max_length: int = 32) -> str:
    """
    Generate a sentence using a pre-trained model from Hugging Face Transformers.

    Args:
        words (str): The words to be included in the sentence.
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
    words = 'moon rabbit forest magic'
    sentence = gen_sentence(words)
    assert isinstance(sentence, str)
    assert all(word in sentence for word in words.split())
    words = 'tree plant ground hole dig'
    sentence = gen_sentence(words)
    assert isinstance(sentence, str)
    assert all(word in sentence for word in words.split())
    return 'All Tests Passed'

# call_test_function_code --------------------

test_gen_sentence()