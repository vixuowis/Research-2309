# requirements_file --------------------

!pip install -U torch transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

# function_code --------------------

def generate_response(user_input, chat_history_ids):
    """Generate a response for the main character controlled by AI.

    Args:
        user_input (str): The input message from the user.
        chat_history_ids (torch.Tensor): Tensor containing the past conversation history.

    Returns:
        str: AI-generated response based on the main character's persona.

    Raises:
        ValueError: If the input is not a string or empty.
    """
    if not user_input or not isinstance(user_input, str):
        raise ValueError("Input must be a non-empty string.")

    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    model = AutoModelWithLMHead.from_pretrained('output-small')
    user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, user_input_ids], dim=-1) if chat_history_ids is not None and chat_history_ids.numel() > 0 else user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature=0.8)
    ai_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return ai_response

# test_function_code --------------------

def test_generate_response():
    print("Testing started.")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')

    # Test case 1: Valid input
    print("Testing case [1/3] started.")
    user_input = "Hello, AI."
    chat_history_ids = None
    response = generate_response(user_input, chat_history_ids)
    assert isinstance(response, str), f"Test case [1/3] failed: Expected string response, got {type(response)}"

    # Test case 2: Empty input
    print("Testing case [2/3] started.")
    user_input = ""
    try:
        response = generate_response(user_input, chat_history_ids)
        assert False, "Test case [2/3] failed: ValueError expected for empty input."
    except ValueError as e:
        assert str(e) == "Input must be a non-empty string.", f"Test case [2/3] failed: Unexpected error message {e}"

    # Test case 3: Continued conversation
    print("Testing case [3/3] started.")
    user_input = "How are you?"
    chat_history_ids = tokenizer.encode("Greetings, human. How can I assist?" + tokenizer.eos_token, return_tensors='pt')
    response = generate_response(user_input, chat_history_ids)
    assert isinstance(response, str), f"Test case [3/3] failed: Expected string response, got {type(response)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_response()