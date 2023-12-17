# requirements_file --------------------

!pip install -U transformers==4.5.0

# function_import --------------------

from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer

# function_code --------------------

def generate_next_word(phrase):
    '''
    Generate the next word for a given incomplete phrase using the DeBERTa model.

    Parameters:
        phrase (str): The incomplete sentence with '<|mask|>' token where the word needs to be predicted.

    Returns:
        str: The word predicted to fill in the mask.
    '''
    # Load the pre-trained DeBERTa model
    mask_model = DebertaV2ForMaskedLM.from_pretrained('microsoft/deberta-v2-xxlarge')
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xxlarge')

    # Encode and process the phrase
    processed = tokenizer(phrase, return_tensors='pt')

    # Get model predictions
    predictions = mask_model(**processed).logits.argmax(dim=-1)

    # Decode the predicted index to the word
    predicted_word = tokenizer.decode(predictions[0], skip_special_tokens=True)
    return predicted_word

# test_function_code --------------------

def test_generate_next_word():
    print("Testing generate_next_word function.")

    # Testing with a known phrase
    phrase = "The dog jumped over the <|mask|>"
    predicted_word = generate_next_word(phrase)
    print(f"Predicted word: {predicted_word}")

    # Expected result is a logical word that fits the phrase (e.g., 'fence')
    assert predicted_word.isalpha(), "The predicted word should only contain alphabetic characters."

    print("All tests passed!")