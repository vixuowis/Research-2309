# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer

# function_code --------------------

def predict_next_word(phrase):
    """
    Predicts the next word in the given phrase using the DeBERTa V2 xxlarge model.

    Args:
        phrase (str): The input phrase with a mask token where the next word should be predicted.

    Returns:
        str: The predicted next word to complete the phrase.
    """
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xxlarge')
    model = DebertaV2ForMaskedLM.from_pretrained('microsoft/deberta-v2-xxlarge')

    # Add the mask token to the phrase
    masked_phrase = phrase.replace('...', '<|mask|>')
    inputs = tokenizer(masked_phrase, return_tensors='pt')
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    predicted_word = tokenizer.decode(predictions[0], skip_special_tokens=True)

    return predicted_word.strip()

# test_function_code --------------------

def test_predict_next_word():
    print("Testing started.")

    # Test case 1
    print("Testing case [1/3] started.")
    predicted_word = predict_next_word("The dog jumped over the ...")
    assert predicted_word.isalpha(), f"Test case [1/3] failed: Predicted word must be alphabetic, got {predicted_word}."

    # Test case 2
    print("Testing case [2/3] started.")
    predicted_word = predict_next_word("A rolling stone gathers no ...")
    assert predicted_word.lower() == 'moss', f"Test case [2/3] failed: Expected 'moss', got {predicted_word}."

    # Test case 3
    print("Testing case [3/3] started.")
    predicted_word = predict_next_word("Actions speak louder than ...")
    assert predicted_word.lower() in ['words', 'actions'], f"Test case [3/3] failed: Expected 'words' or 'actions', got {predicted_word}."
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_next_word()