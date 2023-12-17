# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import DebertaTokenizer, DebertaModel

# function_code --------------------

def improve_readability(sentence: str) -> str:
    """
    Improve the readability and grammaticality of the provided sentence by suggesting the best
    replacement for the masked part.

    Args:
        sentence (str): A sentence containing the '[MASK]' token where the replacement is needed.

    Returns:
        str: The sentence with improved readability where the '[MASK]' token is replaced.

    Raises:
        ValueError: If the '[MASK]' token is not found in the sentence.
    """
    if '[MASK]' not in sentence:
        raise ValueError("The sentence does not contain the '[MASK]' token.")

    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-v2-xlarge')
    model = DebertaModel.from_pretrained('microsoft/deberta-v2-xlarge')
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)
    predictions = outputs.logits
    mask_token_index = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
    predicted_token_id = predictions[:, mask_token_index].argmax(-1).item()
    predicted_token = tokenizer.decode([predicted_token_id])
    improved_sentence = sentence.replace('[MASK]', predicted_token)
    return improved_sentence

# test_function_code --------------------

def test_improve_readability():
    print("Testing started.")

    # Test case 1
    print("Testing case [1/1] started.")
    sentence = "The cat was chasing its [MASK]."
    expected_output = "The cat was chasing its tail."
    assert improve_readability(sentence) == expected_output, f"Test case [1/1] failed: expected {expected_output}"
    print("Testing finished.")

# call_test_function_line --------------------

test_improve_readability()