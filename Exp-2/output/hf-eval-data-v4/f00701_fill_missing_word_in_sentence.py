# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def fill_missing_word_in_sentence(sentence):
    """
    Given a sentence with a missing word indicated by [MASK], this function
    uses Bio_ClinicalBERT to fill in the missing word.
    
    Args:
        sentence (str): The sentence with a missing word, represented by [MASK].

    Returns:
        str: The sentence with the [MASK] replaced by the predicted word.
    """
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    # Encode the sentence
    input_tokens = tokenizer.encode(sentence, return_tensors='pt')

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_tokens)
        predictions = outputs.logits

    # Get the predicted word for [MASK]
    predicted_index = predictions[0, input_tokens[0] == tokenizer.mask_token_id].argmax(axis=-1)
    predicted_word = tokenizer.decode(predicted_index)

    # Replace [MASK] with the predicted word in the sentence
    filled_sentence = sentence.replace('[MASK]', predicted_word)

    return filled_sentence

# test_function_code --------------------

def test_fill_missing_word_in_sentence():
    print("Testing fill_missing_word_in_sentence function.")

    # Test case 1: Sentence with a single mask
    sentence = "The patient showed signs of fever and a [MASK] heart rate."
    expected = "The patient showed signs of fever and a high heart rate."  # Sample expected output
    result = fill_missing_word_in_sentence(sentence)
    assert result == expected, f"Test case 1 failed: expected {{expected}}, got {{result}}"

    # Test case 2: Sentence with multiple masks (only the first mask will be filled)
    sentence = "The [MASK] was administered to the patient suffering from [MASK]."
    expected = "The medication was administered to the patient suffering from [MASK]."  # Sample expected output
    result = fill_missing_word_in_sentence(sentence)
    assert result == expected, f"Test case 2 failed: expected {{expected}}, got {{result}}"

    # Test case 3: Sentence without a mask
    sentence = "Normal vitals with no indications of infection."
    expected = sentence  # The sentence should remain unchanged
    result = fill_missing_word_in_sentence(sentence)
    assert result == expected, f"Test case 3 failed: expected {{expected}}, got {{result}}"

    print("All tests passed.")

test_fill_missing_word_in_sentence()