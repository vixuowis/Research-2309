# requirements_file --------------------

!pip install -U transformers, torch

# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def generate_chatbot_response(user_input, model, tokenizer, chat_history_ids):
    # Encode the new user input, add the eos_token and return a PyTorch tensor
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    # Concatenate the new user input with the chat history (if available). Otherwise, just use the user input
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    # Generate a response from the model
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    # Decode the generated response and skip special tokens
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# test_function_code --------------------

def test_generate_chatbot_response():
    print("Testing generate_chatbot_response function.")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
    chat_history_ids = None

    # Test case 1: Provide an initial user input and expect a response
    print("Test case [1/3] started.")
    response, chat_history_ids = generate_chatbot_response("Hello, how are you?", model, tokenizer, chat_history_ids)
    assert isinstance(response, str), "Test case [1/3] failed: The response should be a string."
    assert chat_history_ids is not None, "Test case [1/3] failed: chat_history_ids should be updated."

    # Test case 2: Provide a follow-up question and expect a response
    print("Test case [2/3] started.")
    response, _ = generate_chatbot_response("What are your working hours?", model, tokenizer, chat_history_ids)
    assert isinstance(response, str), "Test case [2/3] failed: The response should be a string."

    print("All test cases passed!")

# Running the test function
test_generate_chatbot_response()