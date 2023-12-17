# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelWithLMHead, AutoTokenizer

# function_code --------------------

def generate_creative_sentence(words, max_length=32):
    """Generate a creative sentence using specific words.

    Args:
        words (str): The words to be included in the sentence.
        max_length (int): Maximum length of the generated sentence.

    Returns:
        str: A creative sentence containing the input words.

    Raises:
        ValueError: If the words are not provided.
    """
    if not words:
        raise ValueError('No words were provided for sentence generation.')

    tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-common_gen')
    model = AutoModelWithLMHead.from_pretrained('mrm8488/t5-base-finetuned-common_gen')

    input_text = words
    features = tokenizer([input_text], return_tensors='pt')
    output = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'], max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# test_function_code --------------------

def test_generate_creative_sentence():
    print("Testing started.")

    # Test case 1: Check if function returns a string
    print("Testing case [1/3] started.")
    result = generate_creative_sentence('moon rabbit forest magic')
    assert isinstance(result, str), f"Test case [1/3] failed: result is not a string, got {type(result)}"

    # Test case 2: Check if function includes all input words
    print("Testing case [2/3] started.")
    assert all(word in result for word in ['moon', 'rabbit', 'forest', 'magic']), "Test case [2/3] failed: Not all input words are included in the result."

    # Test case 3: Check ValueError for empty input
    print("Testing case [3/3] started.")
    try:
        generate_creative_sentence('')
        assert False, "Test case [3/3] failed: ValueError was not raised for empty words input."
    except ValueError as e:
        assert str(e) == "No words were provided for sentence generation.", f"Test case [3/3] failed: Wrong error message for ValueError, got {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_generate_creative_sentence()