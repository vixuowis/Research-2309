# requirements_file --------------------

!pip install -U transformers accelerate bitsandbytes

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

# function_code --------------------

def translate_french_to_english(sentence: str) -> str:
    """
    Translate a French sentence into English using a pre-trained model.

    Args:
        sentence (str): The French sentence to be translated.

    Returns:
        str: The translated English sentence.

    Raises:
        Exception: If translation fails.
    """
    checkpoint = 'bigscience/bloomz-560m'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    inputs = tokenizer.encode(f'Translate to English: {sentence}', return_tensors='pt')
    outputs = model.generate(inputs)
    return tokenizer.decode(outputs[0]).strip()


# test_function_code --------------------

def test_translate_french_to_english():
    print("Testing started.")

    # Test case 1: Common greeting
    print("Testing case [1/3] started.")
    result = translate_french_to_english('Bonjour')
    assert result == 'Hello', f"Test case [1/3] failed: Expected 'Hello', got '{result}'"

    # Test case 2: Phrase expressing love
    print("Testing case [2/3] started.")
    result = translate_french_to_english('Je t\u2019aime')
    assert result == 'I love you', f"Test case [2/3] failed: Expected 'I love you', got '{result}'"

    # Test case 3: Simple question
    print("Testing case [3/3] started.")
    result = translate_french_to_english('Comment vas-tu?')
    assert result == 'How are you?', f"Test case [3/3] failed: Expected 'How are you?', got '{result}'"
    print("Testing finished.")


# call_test_function_line --------------------

test_translate_french_to_english()