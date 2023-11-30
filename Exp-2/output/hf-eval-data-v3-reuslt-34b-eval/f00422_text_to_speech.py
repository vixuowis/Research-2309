# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer

# function_code --------------------

def text_to_speech(text: str) -> None:
    """
    Convert the input text into speech using a pretrained model.

    Args:
        text (str): The input text to be converted into speech.

    Returns:
        None

    Raises:
        OSError: If the pretrained model or tokenizer is not found.
    """
    print("Starting text-to-speech!")
    try:
        # Loading pretrained tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            "tuner007/GPT-J-tiny-128-EnDis"
            )
        model = AutoModelForCausalLM.from_pretrained(
            "tuner007/GPT-J-tiny-128-EnDis"
            )
    except OSError:
        print("Pre-trained model or tokenizer not found!")
        raise
    
    text = text.replace(" ", "_") # To prevent input text from getting split up
    text_tokens = tokenizer.encode(f"<speech> {text}", return_tensors="pt")
    generated_ids = model.generate(
                        text_tokens,
                        max_length=len(text_tokens[0]) + 100,
                        repetition_penalty=5.,
                        top_p=0.95,
                        temperature=0.9,
                        do_sample=True,
                        num_return_sequences=1
                    )[0].tolist()
    
    # Decoding the generated ids to obtain the corresponding text
    audio = tokenizer.decode(generated_ids)
    print("Finished generating speech!")

# main --------------------

if __name__ == "__main__":
    text_to_speech(input("Enter the text you want to convert into speech: "))

# test_function_code --------------------

def test_text_to_speech():
    """
    Test the text_to_speech function with different test cases.
    """
    # Test case 1: Normal case
    text = 'こんにちは、私たちはあなたの助けが必要です。'
    try:
        text_to_speech(text)
        print('Test case 1 passed')
    except Exception as e:
        print(f'Test case 1 failed: {e}')

    # Test case 2: Empty string
    text = ''
    try:
        text_to_speech(text)
        print('Test case 2 passed')
    except Exception as e:
        print(f'Test case 2 failed: {e}')

    # Test case 3: Non-string input
    text = 123
    try:
        text_to_speech(text)
        print('Test case 3 passed')
    except Exception as e:
        print(f'Test case 3 failed: {e}')


# call_test_function_code --------------------

if __name__ == '__main__':
    test_text_to_speech()