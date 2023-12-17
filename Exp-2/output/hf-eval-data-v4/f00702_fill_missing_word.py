# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForMaskedLM

# function_code --------------------

def fill_missing_word(sentence_with_mask):
    """
    Fills in the missing word in a sentence using a Dutch BERT pre-trained model.
    
    :param sentence_with_mask: A sentence with a missing word indicated by a mask token.
    :return: The sentence with the missing word filled in.
    """
    # Load the pre-trained Dutch BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    model = AutoModelForMaskedLM.from_pretrained('GroNLP/bert-base-dutch-cased')

    # Encode the text, adding special tokens for BERT to work
    input_tokens = tokenizer.encode(sentence_with_mask, return_tensors="pt")

    # Find the index of the mask token
    mask_position = input_tokens.tolist()[0].index(tokenizer.mask_token_id)

    # Predict the missing word
    output = model(input_tokens)
    prediction = output.logits[0, mask_position].argmax().item()

    # Convert the predicted token ID to the actual word
    predicted_word = tokenizer.convert_ids_to_tokens(prediction)

    # Replace the mask with the predicted word and return the sentence
    filled_sentence = sentence_with_mask.replace(tokenizer.mask_token, predicted_word)
    return filled_sentence

# test_function_code --------------------

def test_fill_missing_word():
    print("Testing started.")

    # Test case 1: missing a common noun
    print("Testing case [1/3] started.")
    sentence1 = "Het is vandaag erg koud, dus vergeet niet je [MASK] mee te nemen."
    result1 = fill_missing_word(sentence1)
    assert "[MASK]" not in result1, f"Test case [1/3] failed: Mask not filled. Result: {result1}"

    # Test case 2: missing a verb
    print("Testing case [2/3] started.")
    sentence2 = "Zij [MASK] naar de winkel elke maandag."
    result2 = fill_missing_word(sentence2)
    assert "[MASK]" not in result2, f"Test case [2/3] failed: Mask not filled. Result: {result2}"

    # Test case 3: missing an adjective
    print("Testing case [3/3] started.")
    sentence3 = "De kat is [MASK] dan de hond."
    result3 = fill_missing_word(sentence3)
    assert "[MASK]" not in result3, f"Test case [3/3] failed: Mask not filled. Result: {result3}"

    print("Testing finished.")

# Run the test function
test_fill_missing_word()