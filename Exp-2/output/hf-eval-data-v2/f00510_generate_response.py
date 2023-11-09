# function_import --------------------

from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

# function_code --------------------

def generate_response(user_input):
    """
    Generate a response from the AI based on the user's input.

    Args:
        user_input (str): The input provided by the user.

    Returns:
        str: The AI's response.
    """
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    model = AutoModelWithLMHead.from_pretrained('output-small')

    user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature = 0.8)
    ai_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return ai_response

# test_function_code --------------------

def test_generate_response():
    """
    Test the generate_response function.
    """
    user_input = 'Hello, AI!'
    response = generate_response(user_input)
    assert isinstance(response, str), 'The response should be a string.'

# call_test_function_code --------------------

test_generate_response()