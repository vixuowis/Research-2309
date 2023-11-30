# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

# function_code --------------------

def generate_code(description: str) -> str:
    '''
    Generate code snippets based on natural language descriptions.

    Args:
        description (str): The natural language description.

    Returns:
        str: The generated code snippet.
    '''
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    # encode the new user input
    chat_history_ids = tokenizer.encode(description, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids], dim=-1) if tokenizer is not None else None

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# test_function_code --------------------

def test_generate_code():
    '''
    Test the generate_code function.
    '''
    description1 = 'Write a Python function to calculate the factorial of a number.'
    description2 = 'Write a Python function to sort a list of numbers in ascending order.'
    description3 = 'Write a Python function to reverse a string.'
    assert isinstance(generate_code(description1), str)
    assert isinstance(generate_code(description2), str)
    assert isinstance(generate_code(description3), str)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_code()