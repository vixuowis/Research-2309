# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def generate_dialogue(user_input):
    """
    Generate a dialogue response using DialoGPT-large model.

    Args:
        user_input (str): The user's input to which the chatbot should respond.

    Returns:
        str: The chatbot's response.

    Raises:
        OSError: If there is an issue with loading the pre-trained model or tokenizer.
    """
    # Get the DialoGPT-large model and tokenizer.
    try:
        print("Trying to load DialoGPT-large model...")
        model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')
        tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large', cache_dir='./cache', use_fast=False)
    except OSError as e:
        print("OSError: {}".format(e))
        raise Exception("Failed to load DialoGPT-large model. Please check your internet connection.") from e
    # Get the device to be used for inference.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using {} as inference device.".format(device))
    model.to(device)
    # Set up the inputs to generate a response for.
    bot_input_ids = tokenizer([user_input], return_tensors='pt').input_ids
    bot_input_ids = bot_input_ids.to(device)
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)[0]
    return tokenizer.decode(chat_history_ids, skip_special_tokens=True)

# test_function_code --------------------

def test_generate_dialogue():
    """
    Test the generate_dialogue function.
    """
    test_input = 'How do I search for scientific papers?'
    response = generate_dialogue(test_input)
    assert isinstance(response, str), 'The response should be a string.'

    test_input = 'What is the weather like today?'
    response = generate_dialogue(test_input)
    assert isinstance(response, str), 'The response should be a string.'

    test_input = 'Tell me a joke.'
    response = generate_dialogue(test_input)
    assert isinstance(response, str), 'The response should be a string.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_dialogue()