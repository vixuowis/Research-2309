# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForMaskedLM

# function_code --------------------

def fill_in_missing_words(masked_text: str) -> str:
    '''
    Fill in missing words in a Japanese text using a pre-trained BERT model.

    Args:
        masked_text (str): The text with a '[MASK]' token where the word is missing.

    Returns:
        str: The text with the missing word filled in by the BERT model.

    Raises:
        ValueError: If 'masked_text' does not contain '[MASK]'.
    '''
    if '[MASK]' not in masked_text:
        raise ValueError("The input must contain a '[MASK]' token.")

    tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    model = AutoModelForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese')

    encoded_input = tokenizer(masked_text, return_tensors='pt')
    outputs = model(**encoded_input)

    prediction = outputs.logits.argmax(-1)
    predicted_token = tokenizer.convert_ids_to_tokens(prediction[0])

    filled_text = masked_text.replace('[MASK]', predicted_token[1])

    return filled_text

# test_function_code --------------------

def test_fill_in_missing_words():
    print("Testing started.")

    # Test case 1: Input with one masked token
    print("Testing case [1/3] started.")
    input_text = '日本の首都は[MASK]です。'
    expected_output = '日本の首都は東京です。'
    actual_output = fill_in_missing_words(input_text)
    assert actual_output == expected_output, f"Test case [1/3] failed: Expected \'{expected_output}\' but got \'{actual_output}\'."

    # Test case 2: Input with no masked token raises an error
    print("Testing case [2/3] started.")
    input_text = 'この文章にはマスクされた語が含まれていません。'
    try:
        fill_in_missing_words(input_text)
        assert False, "Test case [2/3] failed: Expected ValueError was not raised."
    except ValueError as e:
        assert str(e) == "The input must contain a '[MASK]' token.", f"Test case [2/3] failed: Unexpected error message {str(e)}."

    # Test case 3: Input with multiple masked tokens should fill in the first one
    print("Testing case [3/3] started.")
    input_text = 'トンボの[MASK]は[MASK]に住んでいます。'
    expected_output = 'トンボの眼は水辺に住んでいます。'  # Assuming the model fills '眼' for the first '[MASK]'
    actual_output = fill_in_missing_words(input_text)
    assert actual_output == expected_output, f"Test case [3/3] failed: Expected \'{expected_output}\' but got \'{actual_output}\'."

    print("Testing finished.")

# call_test_function_line --------------------

test_fill_in_missing_words()