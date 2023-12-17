# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForMaskedLM

# function_code --------------------

def complete_dutch_sentence(sentence:str) -> str:
    '''
    Complete a Dutch sentence by filling in the masked word ([MASK]) using BERTje model.

    Args:
    sentence (str): The Dutch sentence with a [MASK] token where the word should be predicted.

    Returns:
    str: The completed sentence with the masked word filled in.
    '''
    # Initialize the tokenizer and model using the Dutch BERTje model
    tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    model = AutoModelForMaskedLM.from_pretrained('GroNLP/bert-base-dutch-cased')

    # Tokenize the input sentence and retrieve the masked index
    input_tokens = tokenizer(sentence, return_tensors='pt')
    mask_token_index = input_tokens.input_ids[0].tolist().index(tokenizer.mask_token_id)

    # Predict the token for the masked position
    with torch.no_grad():
        outputs = model(**input_tokens)
    predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1).item()

    # Replace the mask token with the predicted word
    predicted_word = tokenizer.decode([predicted_token_id])
    completed_sentence = sentence.replace(tokenizer.mask_token, predicted_word)

    return completed_sentence


# test_function_code --------------------

def test_complete_dutch_sentence():
    print("Testing started.")

    # Test case 1: Typical sentence with one masked word
    sentence = "Hij ging naar de [MASK] om boodschappen te doen."
    completed_sentence = complete_dutch_sentence(sentence)
    print("Test case [1/1] started.")
    assert '[MASK]' not in completed_sentence, f"Test case [1/1] failed: Mask not filled. Completed sentence: {completed_sentence}"

    # There's no direct way to test the correctness of predicted word as it could vary.
    # However, this test ensures that the model runs and fills the mask.
    print("Test case [1/1] passed. Testing finished.")

# Run the test function
test_complete_dutch_sentence()
