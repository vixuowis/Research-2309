# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# function_code --------------------

def complete_dutch_sentence(sentence: str) -> str:
    """
    Complete a Dutch sentence by filling in the masked word using a pre-trained model.

    Args:
        sentence (str): A Dutch sentence with a [MASK] token where the word should be predicted.

    Returns:
        str: The sentence with the [MASK] replaced by the predicted word.

    Raises:
        ValueError: If the input sentence does not contain a [MASK] token.
    """
    if '[MASK]' not in sentence:
        raise ValueError('The input sentence must contain a [MASK] token.')

    tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    model = AutoModelForMaskedLM.from_pretrained('GroNLP/bert-base-dutch-cased')
    input_tokens = tokenizer(sentence, return_tensors='pt')
    mask_token_index = torch.where(input_tokens['input_ids'] == tokenizer.mask_token_id)[1]
    output = model(**input_tokens)
    predicted_token_id = torch.argmax(output.logits[0, mask_token_index]).item()
    predicted_word = tokenizer.decode([predicted_token_id])
    completed_sentence = sentence.replace('[MASK]', predicted_word)
    return completed_sentence

# test_function_code --------------------

def test_complete_dutch_sentence():
    print("Testing started.")
    test_cases = [
        ("Hij ging naar de [MASK] om boodschappen te doen.", "winkel"),
        ("Zij heeft [MASK] mooie bloemen in de tuin.", "enkele")
    ]

    for i, (sentence, expected) in enumerate(test_cases, start=1):
        print(f"Testing case [{i}/{len(test_cases)}] started.")
        completed_sentence = complete_dutch_sentence(sentence)
        assert expected in completed_sentence, f"Test case [{i}/{len(test_cases)}] failed: Expected word not found in completed sentence."
    print("Testing finished.")

# call_test_function_line --------------------

test_complete_dutch_sentence()