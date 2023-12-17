# requirements_file --------------------

!pip install -U transformers, torch

# function_import --------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# function_code --------------------

def start_conversation(initial_message):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
    chat_history = None
    user_input = initial_message

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    chat_history = torch.cat([chat_history, input_ids], dim=-1) if chat_history is not None else input_ids
    outputs = model.generate(chat_history, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response

# test_function_code --------------------

def test_start_conversation():
    print("Testing start_conversation function.")
    initial_message = "Hello, how are you?"

    response = start_conversation(initial_message)

    assert isinstance(response, str), f"Expected string response, got: {type(response)}"
    assert len(response) > 0, "Response is empty."

    print("Test passed successfully.")